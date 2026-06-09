"""Suno API Client for music generation."""

import asyncio
import httpx
import os
import re
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from callback_server import resolve_callback_url, validate_callback_url
from suno_response import (
    extract_generation_task_id,
    extract_lyrics_from_container,
    extract_tracks_from_container,
    is_lyrics_failure,
    is_lyrics_success,
    is_processing_failure,
    is_processing_success,
    is_terminal_failure,
    is_terminal_success,
)


class SunoAPIError(Exception):
    """Base exception for Suno API errors."""
    pass


MODEL_VERSION_MAP = {
    "v4": "V4",
    "v4.5": "V4_5",
    "v4.5plus": "V4_5PLUS",
    "v4.5all": "V4_5ALL",
    "v5": "V5",
    "v5.5": "V5_5",
}

DEFAULT_MODEL_VERSION = "V5"


class SunoClient:
    """Client for interacting with the Suno API."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the Suno API client.

        Args:
            api_key: Suno API key. If not provided, will read from SUNO_API_KEY env var.
            base_url: Base URL for the Suno API. Defaults to https://api.sunoapi.org
        """
        self.api_key = api_key or os.getenv("SUNO_API_KEY")
        if not self.api_key:
            raise ValueError("SUNO_API_KEY must be provided or set in environment")

        self.base_url = base_url or os.getenv("SUNO_API_BASE_URL", "https://api.sunoapi.org")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=300.0  # 5 minute timeout for music generation
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    @staticmethod
    def map_model_version(model_version: str) -> str:
        return MODEL_VERSION_MAP.get(model_version.lower(), DEFAULT_MODEL_VERSION)

    @staticmethod
    def resolve_callback_url(callback_url: Optional[str] = None) -> str:
        url = resolve_callback_url(callback_url)
        validate_callback_url(url)
        return url

    @staticmethod
    def _raise_api_error(result: Dict[str, Any], context: str) -> None:
        code = result.get("code")
        if code == 200:
            return

        error_msg = result.get("msg", "Unknown error")
        if code == 430:
            raise SunoAPIError(
                f"{context}: rate limit exceeded (code 430). "
                f"Reduce request frequency and retry. Details: {error_msg}"
            )
        if code == 429:
            raise SunoAPIError(f"{context}: insufficient credits (code 429). Details: {error_msg}")
        raise SunoAPIError(f"{context}: API error (code {code}): {error_msg}")

    @staticmethod
    def _apply_advanced_music_params(
        payload: Dict[str, Any],
        persona_id: Optional[str] = None,
        persona_model: Optional[str] = None,
        negative_tags: Optional[str] = None,
        vocal_gender: Optional[str] = None,
        style_weight: Optional[float] = None,
        weirdness_constraint: Optional[float] = None,
        audio_weight: Optional[float] = None,
    ) -> None:
        if persona_id:
            payload["personaId"] = persona_id
        if persona_model in ("style_persona", "voice_persona"):
            payload["personaModel"] = persona_model
        if negative_tags:
            payload["negativeTags"] = negative_tags
        if vocal_gender in ("m", "f"):
            payload["vocalGender"] = vocal_gender
        if style_weight is not None and 0.0 <= style_weight <= 1.0:
            payload["styleWeight"] = style_weight
        if weirdness_constraint is not None and 0.0 <= weirdness_constraint <= 1.0:
            payload["weirdnessConstraint"] = weirdness_constraint
        if audio_weight is not None and 0.0 <= audio_weight <= 1.0:
            payload["audioWeight"] = audio_weight

    @staticmethod
    def _validate_generation_task_and_audio_ids(task_id: str, audio_id: str) -> None:
        if not task_id:
            raise ValueError("task_id is required (generation job ID)")
        if not audio_id:
            raise ValueError("audio_id is required (track UUID)")

        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        if not uuid_pattern.match(audio_id):
            raise ValueError(
                f"Invalid audio_id format: '{audio_id}'. Expected UUID from track results."
            )
        if uuid_pattern.match(task_id):
            raise ValueError(
                f"task_id '{task_id}' looks like a track UUID. Use the generation task ID instead."
            )

    async def _post(
        self,
        path: str,
        payload: Dict[str, Any],
        context: str,
        allowed_codes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        try:
            response = await self.client.post(path, json=payload)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict):
                code = result.get("code")
                if code != 200 and code not in (allowed_codes or []):
                    self._raise_api_error(result, context)
            return result
        except SunoAPIError:
            raise
        except httpx.HTTPError as exc:
            raise SunoAPIError(f"{context}: {exc}") from exc

    async def _get(self, path: str, params: Dict[str, Any], context: str) -> Dict[str, Any]:
        try:
            response = await self.client.get(path, params=params)
            response.raise_for_status()
            result = response.json()
            if isinstance(result, dict):
                self._raise_api_error(result, context)
            return result
        except httpx.HTTPError as exc:
            raise SunoAPIError(f"{context}: {exc}") from exc

    async def generate_music(
        self,
        prompt: Optional[str] = None,
        make_instrumental: bool = False,
        model_version: str = DEFAULT_MODEL_VERSION,
        custom_mode: bool = False,
        style: Optional[str] = None,
        title: Optional[str] = None,
        callback_url: Optional[str] = None,
        persona_id: Optional[str] = None,
        persona_model: Optional[str] = None,
        negative_tags: Optional[str] = None,
        vocal_gender: Optional[str] = None,
        style_weight: Optional[float] = None,
        weirdness_constraint: Optional[float] = None,
        audio_weight: Optional[float] = None,
        wait_audio: bool = False,
        callback_store: Optional[Any] = None,
        poll_interval: float = 10.0,
        max_wait: float = 600.0,
    ) -> Dict[str, Any]:
        """
        Generate music from a text prompt.

        Args:
            prompt: Text description/lyrics for the music.
            make_instrumental: If True, generate instrumental only (no vocals)
            model_version: AI model version (V4, V4_5, V4_5PLUS, V4_5ALL, V5, V5_5)
            custom_mode: If True, use custom mode (requires style and title; prompt required if not instrumental)
            style: Music style/genre (required in Custom Mode)
            title: Song title (required in Custom Mode)
            callback_url: Webhook URL for completion notification (auto-resolved if omitted)
            persona_id: Persona identifier for stylistic influence (Custom Mode only)
            persona_model: Persona type (`style_persona` or `voice_persona`)
            negative_tags: Styles/traits to exclude from generation
            vocal_gender: Preferred vocal gender ('m' or 'f')
            style_weight: Weight of style guidance (0.00-1.00)
            weirdness_constraint: Creative deviation tolerance (0.00-1.00)
            audio_weight: Input audio influence weighting (0.00-1.00)
            wait_audio: Poll until generation completes before returning
            callback_store: Optional callback store for webhook-driven completion
            poll_interval: Seconds between status polls when wait_audio is True
            max_wait: Maximum seconds to wait for generation

        Returns:
            Dictionary containing generation task info and optional completed tracks

        Raises:
            SunoAPIError: If the API request fails
            ValueError: If required parameters are missing or invalid
        """
        if custom_mode:
            if not style:
                raise ValueError("style must be provided when custom_mode is True")
            if not title:
                raise ValueError("title must be provided when custom_mode is True")
            if not make_instrumental and not prompt:
                raise ValueError("prompt must be provided when custom_mode is True and instrumental is False")

        if not custom_mode and not prompt:
            raise ValueError("prompt is required in non-custom mode")

        resolved_callback_url = self.resolve_callback_url(callback_url)

        payload: Dict[str, Any] = {
            "instrumental": make_instrumental,
            "model": model_version,
            "customMode": custom_mode,
            "callBackUrl": resolved_callback_url,
        }

        if prompt:
            payload["prompt"] = prompt

        if custom_mode:
            payload["style"] = style
            payload["title"] = title

        self._apply_advanced_music_params(
            payload,
            persona_id=persona_id,
            persona_model=persona_model,
            negative_tags=negative_tags,
            vocal_gender=vocal_gender,
            style_weight=style_weight,
            weirdness_constraint=weirdness_constraint,
            audio_weight=audio_weight,
        )

        try:
            result = await self._post("/api/v1/generate", payload, "Failed to generate music")

            if wait_audio:
                task_id = extract_generation_task_id(result)
                if not task_id:
                    raise SunoAPIError("Generation started but no taskId was returned")

                completed = await self.wait_for_task_completion(
                    task_id=task_id,
                    callback_store=callback_store,
                    poll_interval=poll_interval,
                    max_wait=max_wait,
                )
                completed_data = completed.get("data", {})
                tracks = extract_tracks_from_container(completed_data)
                result = {
                    **result,
                    "data": {
                        "taskId": task_id,
                        "status": completed_data.get("status"),
                        "tracks": tracks,
                    },
                }

            return result
        except SunoAPIError:
            raise
        except Exception as exc:
            raise SunoAPIError(f"Failed to generate music: {exc}") from exc

    async def wait_for_task_completion(
        self,
        task_id: str,
        callback_store: Optional[Any] = None,
        poll_interval: float = 10.0,
        max_wait: float = 600.0,
    ) -> Dict[str, Any]:
        """Poll task status until generation completes or fails."""
        deadline = time.monotonic() + max_wait

        while time.monotonic() < deadline:
            if callback_store is not None:
                callback_payload = callback_store.get(task_id)
                if callback_payload is not None:
                    callback_data = callback_payload.get("data", {})
                    callback_type = (callback_data.get("callbackType") or "").lower()
                    if callback_type in ("complete", "first"):
                        status_result = await self.get_task_status(task_id)
                        status_data = status_result.get("data", {})
                        if is_terminal_success(status_data.get("status"), extract_tracks_from_container(status_data)):
                            return status_result

            result = await self.get_task_status(task_id)
            data = result.get("data", {})
            status = data.get("status")
            tracks = extract_tracks_from_container(data)

            if is_terminal_failure(status):
                error_message = data.get("errorMessage") or data.get("msg") or status
                raise SunoAPIError(f"Generation failed: {error_message}")

            if is_terminal_success(status, tracks):
                return result

            await asyncio.sleep(poll_interval)

        raise SunoAPIError(f"Generation timeout after {max_wait} seconds for task {task_id}")

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a music generation task using taskId.

        Args:
            task_id: Task ID returned from generate_music

        Returns:
            Dictionary containing task status and track information

        Raises:
            SunoAPIError: If the API request fails
        """
        return await self._get(
            "/api/v1/generate/record-info",
            {"taskId": task_id},
            "Failed to get task status",
        )

    async def get_music_info(self, ids: List[str]) -> Dict[str, Any]:
        """
        Get information about generated music tracks using track IDs.

        Args:
            ids: List of track IDs to retrieve information for

        Returns:
            Dictionary containing track information

        Raises:
            SunoAPIError: If the API request fails
        """
        ids_param = ",".join(ids)
        return await self._get(
            "/api/v1/generate/record-info",
            {"ids": ids_param},
            "Failed to get music info",
        )

    async def get_credits(self) -> Dict[str, Any]:
        """
        Get remaining API credits.

        Returns:
            Dictionary containing credit balance and usage statistics

        Raises:
            SunoAPIError: If the API request fails
        """
        return await self._get("/api/v1/generate/credit", {}, "Failed to get credits")

    async def convert_to_wav(
        self,
        task_id: str,
        audio_id: str,
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert a generated audio track to WAV format.

        Per Suno API docs: BOTH taskId AND audioId are REQUIRED parameters.
        The audioId specifies which exact track to convert (tasks can have multiple tracks).

        Args:
            callback_url: Webhook URL for conversion completion notification (required)
            task_id: Original generation task ID (taskId) from generate_music (required)
            audio_id: Track ID (audioId) of the specific track to convert (required)

        Returns:
            Dictionary containing WAV conversion task information with taskId

        Raises:
            SunoAPIError: If the API request fails
            ValueError: If required parameters are missing or invalid
        """
        resolved_callback_url = self.resolve_callback_url(callback_url)

        self._validate_generation_task_and_audio_ids(task_id, audio_id)

        payload: Dict[str, Any] = {
            "callBackUrl": resolved_callback_url,
            "taskId": task_id,
            "audioId": audio_id,
        }

        result = await self._post(
            "/api/v1/wav/generate",
            payload,
            "Failed to convert to WAV",
            allowed_codes=[409],
        )
        if result.get("code") == 409:
            raise SunoAPIError(
                f"WAV conversion already exists for this track (code 409). "
                f"Use the original conversion task_id with get_wav_conversion_status."
            )
        return result

    async def get_wav_conversion_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a WAV conversion task using taskId.

        Args:
            task_id: Task ID returned from convert_to_wav

        Returns:
            Dictionary containing WAV conversion status and download information

        Raises:
            SunoAPIError: If the API request fails
            ValueError: If task_id is missing
        """
        if not task_id:
            raise ValueError("task_id is required to check WAV conversion status")

        return await self._get(
            "/api/v1/wav/record-info",
            {"taskId": task_id},
            "Failed to get WAV conversion status",
        )

    async def generate_lyrics(
        self,
        prompt: str,
        callback_url: Optional[str] = None,
        wait_completion: bool = True,
        callback_store: Optional[Any] = None,
        poll_interval: float = 5.0,
        max_wait: float = 120.0,
    ) -> Dict[str, Any]:
        if not prompt:
            raise ValueError("prompt is required for lyrics generation")
        if len(prompt) > 200:
            raise ValueError("prompt must be 200 characters or fewer for lyrics generation")

        payload = {
            "prompt": prompt,
            "callBackUrl": self.resolve_callback_url(callback_url),
        }
        result = await self._post("/api/v1/lyrics", payload, "Failed to generate lyrics")

        if wait_completion:
            task_id = extract_generation_task_id(result)
            if not task_id:
                raise SunoAPIError("Lyrics generation started but no taskId was returned")
            completed = await self.wait_for_lyrics_completion(
                task_id=task_id,
                callback_store=callback_store,
                poll_interval=poll_interval,
                max_wait=max_wait,
            )
            completed_data = completed.get("data", {})
            result = {
                **result,
                "data": {
                    "taskId": task_id,
                    "status": completed_data.get("status"),
                    "lyrics": extract_lyrics_from_container(completed_data),
                },
            }
        return result

    async def wait_for_lyrics_completion(
        self,
        task_id: str,
        callback_store: Optional[Any] = None,
        poll_interval: float = 5.0,
        max_wait: float = 120.0,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            if callback_store is not None and callback_store.get(task_id):
                callback_type = (callback_store.get(task_id).get("data", {}).get("callbackType") or "").lower()
                if callback_type == "complete":
                    return await self.get_lyrics_status(task_id)

            result = await self.get_lyrics_status(task_id)
            data = result.get("data", {})
            status = data.get("status")
            lyrics = extract_lyrics_from_container(data)

            if is_lyrics_failure(status):
                raise SunoAPIError(
                    f"Lyrics generation failed: {data.get('errorMessage') or status}"
                )
            if is_lyrics_success(status, lyrics):
                return result

            await asyncio.sleep(poll_interval)

        raise SunoAPIError(f"Lyrics generation timeout after {max_wait} seconds for task {task_id}")

    async def get_lyrics_status(self, task_id: str) -> Dict[str, Any]:
        if not task_id:
            raise ValueError("task_id is required")
        return await self._get(
            "/api/v1/lyrics/record-info",
            {"taskId": task_id},
            "Failed to get lyrics status",
        )

    async def extend_music(
        self,
        audio_id: str,
        model_version: str = DEFAULT_MODEL_VERSION,
        default_param_flag: bool = True,
        prompt: Optional[str] = None,
        style: Optional[str] = None,
        title: Optional[str] = None,
        continue_at: Optional[float] = None,
        callback_url: Optional[str] = None,
        persona_id: Optional[str] = None,
        persona_model: Optional[str] = None,
        negative_tags: Optional[str] = None,
        vocal_gender: Optional[str] = None,
        style_weight: Optional[float] = None,
        weirdness_constraint: Optional[float] = None,
        audio_weight: Optional[float] = None,
        wait_audio: bool = False,
        callback_store: Optional[Any] = None,
        poll_interval: float = 10.0,
        max_wait: float = 600.0,
    ) -> Dict[str, Any]:
        if not audio_id:
            raise ValueError("audio_id is required")

        if default_param_flag:
            if not prompt or not style or not title or continue_at is None:
                raise ValueError(
                    "prompt, style, title, and continue_at are required when default_param_flag is true"
                )

        payload: Dict[str, Any] = {
            "audioId": audio_id,
            "defaultParamFlag": default_param_flag,
            "model": model_version,
            "callBackUrl": self.resolve_callback_url(callback_url),
        }
        if default_param_flag:
            payload.update({
                "prompt": prompt,
                "style": style,
                "title": title,
                "continueAt": continue_at,
            })
        self._apply_advanced_music_params(
            payload,
            persona_id=persona_id,
            persona_model=persona_model,
            negative_tags=negative_tags,
            vocal_gender=vocal_gender,
            style_weight=style_weight,
            weirdness_constraint=weirdness_constraint,
            audio_weight=audio_weight,
        )

        result = await self._post("/api/v1/generate/extend", payload, "Failed to extend music")
        if wait_audio:
            task_id = extract_generation_task_id(result)
            if not task_id:
                raise SunoAPIError("Extension started but no taskId was returned")
            completed = await self.wait_for_task_completion(
                task_id=task_id,
                callback_store=callback_store,
                poll_interval=poll_interval,
                max_wait=max_wait,
            )
            completed_data = completed.get("data", {})
            result = {
                **result,
                "data": {
                    "taskId": task_id,
                    "status": completed_data.get("status"),
                    "tracks": extract_tracks_from_container(completed_data),
                },
            }
        return result

    async def separate_vocals(
        self,
        task_id: str,
        audio_id: str,
        separation_type: str = "separate_vocal",
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._validate_generation_task_and_audio_ids(task_id, audio_id)
        if separation_type not in ("separate_vocal", "split_stem"):
            raise ValueError("separation_type must be 'separate_vocal' or 'split_stem'")

        payload = {
            "taskId": task_id,
            "audioId": audio_id,
            "type": separation_type,
            "callBackUrl": self.resolve_callback_url(callback_url),
        }
        return await self._post("/api/v1/vocal-removal/generate", payload, "Failed to separate vocals")

    async def get_vocal_separation_status(self, task_id: str) -> Dict[str, Any]:
        if not task_id:
            raise ValueError("task_id is required")
        return await self._get(
            "/api/v1/vocal-removal/record-info",
            {"taskId": task_id},
            "Failed to get vocal separation status",
        )

    async def create_music_video(
        self,
        task_id: str,
        audio_id: str,
        callback_url: Optional[str] = None,
        author: Optional[str] = None,
        domain_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._validate_generation_task_and_audio_ids(task_id, audio_id)
        payload: Dict[str, Any] = {
            "taskId": task_id,
            "audioId": audio_id,
            "callBackUrl": self.resolve_callback_url(callback_url),
        }
        if author:
            payload["author"] = author[:50]
        if domain_name:
            payload["domainName"] = domain_name[:50]

        result = await self._post(
            "/api/v1/mp4/generate",
            payload,
            "Failed to create music video",
            allowed_codes=[409],
        )
        if result.get("code") == 409:
            raise SunoAPIError(
                "Music video already exists for this track (code 409). "
                "Use get_music_video_status with the original video task_id."
            )
        return result

    async def get_music_video_status(self, task_id: str) -> Dict[str, Any]:
        if not task_id:
            raise ValueError("task_id is required")
        return await self._get(
            "/api/v1/mp4/record-info",
            {"taskId": task_id},
            "Failed to get music video status",
        )

    async def generate_persona(
        self,
        task_id: str,
        audio_id: str,
        name: str,
        description: str,
        style: Optional[str] = None,
        vocal_start: Optional[float] = None,
        vocal_end: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._validate_generation_task_and_audio_ids(task_id, audio_id)
        if not name:
            raise ValueError("name is required")
        if not description:
            raise ValueError("description is required")

        payload: Dict[str, Any] = {
            "taskId": task_id,
            "audioId": audio_id,
            "name": name,
            "description": description,
        }
        if style:
            payload["style"] = style
        if vocal_start is not None:
            payload["vocalStart"] = vocal_start
        if vocal_end is not None:
            payload["vocalEnd"] = vocal_end

        result = await self._post(
            "/api/v1/generate/generate-persona",
            payload,
            "Failed to generate persona",
            allowed_codes=[409],
        )
        if result.get("code") == 409:
            raise SunoAPIError("Persona already exists for this audio_id (code 409).")
        return result

    async def upload_and_cover_audio(
        self,
        upload_url: str,
        model_version: str = DEFAULT_MODEL_VERSION,
        custom_mode: bool = False,
        make_instrumental: bool = False,
        prompt: Optional[str] = None,
        style: Optional[str] = None,
        title: Optional[str] = None,
        callback_url: Optional[str] = None,
        persona_id: Optional[str] = None,
        persona_model: Optional[str] = None,
        negative_tags: Optional[str] = None,
        vocal_gender: Optional[str] = None,
        style_weight: Optional[float] = None,
        weirdness_constraint: Optional[float] = None,
        audio_weight: Optional[float] = None,
        wait_audio: bool = False,
        callback_store: Optional[Any] = None,
        poll_interval: float = 10.0,
        max_wait: float = 600.0,
    ) -> Dict[str, Any]:
        if not upload_url:
            raise ValueError("upload_url is required")

        if custom_mode:
            if not style or not title:
                raise ValueError("style and title are required when custom_mode is true")
            if not make_instrumental and not prompt:
                raise ValueError("prompt is required when custom_mode is true and make_instrumental is false")
        elif not prompt:
            raise ValueError("prompt is required in non-custom mode")

        payload: Dict[str, Any] = {
            "uploadUrl": upload_url,
            "customMode": custom_mode,
            "instrumental": make_instrumental,
            "model": model_version,
            "callBackUrl": self.resolve_callback_url(callback_url),
        }
        if prompt:
            payload["prompt"] = prompt
        if custom_mode:
            payload["style"] = style
            payload["title"] = title
        self._apply_advanced_music_params(
            payload,
            persona_id=persona_id,
            persona_model=persona_model,
            negative_tags=negative_tags,
            vocal_gender=vocal_gender,
            style_weight=style_weight,
            weirdness_constraint=weirdness_constraint,
            audio_weight=audio_weight,
        )

        result = await self._post(
            "/api/v1/generate/upload-cover",
            payload,
            "Failed to upload and cover audio",
        )
        if wait_audio:
            task_id = extract_generation_task_id(result)
            if not task_id:
                raise SunoAPIError("Upload-cover started but no taskId was returned")
            completed = await self.wait_for_task_completion(
                task_id=task_id,
                callback_store=callback_store,
                poll_interval=poll_interval,
                max_wait=max_wait,
            )
            completed_data = completed.get("data", {})
            result = {
                **result,
                "data": {
                    "taskId": task_id,
                    "status": completed_data.get("status"),
                    "tracks": extract_tracks_from_container(completed_data),
                },
            }
        return result
