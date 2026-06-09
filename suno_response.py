"""Helpers for normalizing Suno API response shapes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


TERMINAL_SUCCESS_STATUSES = {
    "SUCCESS",
    "TEXT_SUCCESS",
    "FIRST_SUCCESS",
    "COMPLETE",
}

TERMINAL_FAILURE_STATUSES = {
    "FAILED",
    "ERROR",
    "CREATE_TASK_FAILED",
}

LYRICS_SUCCESS_STATUSES = {"SUCCESS"}
LYRICS_FAILURE_STATUSES = {
    "CREATE_TASK_FAILED",
    "GENERATE_LYRICS_FAILED",
    "CALLBACK_EXCEPTION",
    "SENSITIVE_WORD_ERROR",
}

PROCESSING_SUCCESS_FLAGS = {"SUCCESS"}
PROCESSING_FAILURE_FLAGS = {
    "CREATE_TASK_FAILED",
    "GENERATE_AUDIO_FAILED",
    "GENERATE_MP4_FAILED",
    "CALLBACK_EXCEPTION",
}


def normalize_track(track: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize camelCase and snake_case track fields."""
    return {
        "id": track.get("id"),
        "title": track.get("title"),
        "status": track.get("status"),
        "model_name": track.get("model_name") or track.get("modelName"),
        "audio_url": track.get("audio_url") or track.get("audioUrl"),
        "stream_audio_url": track.get("stream_audio_url") or track.get("streamAudioUrl"),
        "video_url": track.get("video_url") or track.get("videoUrl"),
        "image_url": track.get("image_url") or track.get("imageUrl"),
        "duration": track.get("duration"),
        "tags": track.get("tags"),
        "prompt": track.get("prompt"),
        "create_time": track.get("create_time") or track.get("createTime"),
        "created_at": track.get("created_at") or track.get("createTime"),
    }


def extract_tracks_from_container(container: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract tracks from task status or callback payloads."""
    if not container or not isinstance(container, dict):
        return []

    response = container.get("response")
    if isinstance(response, dict):
        raw_tracks = response.get("sunoData") or response.get("data") or []
        if raw_tracks:
            return [normalize_track(track) for track in raw_tracks]

    raw_tracks = container.get("data")
    if isinstance(raw_tracks, list):
        return [normalize_track(track) for track in raw_tracks]

    return []


def extract_task_id_from_callback(payload: Dict[str, Any]) -> Optional[str]:
    """Extract task ID from a webhook payload."""
    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    return data.get("task_id") or data.get("taskId")


def extract_callback_type(payload: Dict[str, Any]) -> Optional[str]:
    data = payload.get("data")
    if isinstance(data, dict):
        callback_type = data.get("callbackType")
        if isinstance(callback_type, str):
            return callback_type.lower()
    return None


def is_terminal_success(status: Optional[str], tracks: List[Dict[str, Any]]) -> bool:
    normalized = (status or "").upper()
    if normalized in TERMINAL_SUCCESS_STATUSES:
        return True
    return bool(tracks) and any(track.get("audio_url") for track in tracks)


def is_terminal_failure(status: Optional[str]) -> bool:
    return (status or "").upper() in TERMINAL_FAILURE_STATUSES


def extract_generation_task_id(result: Dict[str, Any]) -> Optional[str]:
    data = result.get("data")
    if isinstance(data, dict):
        return data.get("taskId") or data.get("task_id")
    return None


def extract_lyrics_from_container(container: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not container or not isinstance(container, dict):
        return []

    response = container.get("response")
    if isinstance(response, dict):
        raw_lyrics = response.get("data")
        if isinstance(raw_lyrics, list):
            return [
                {
                    "title": item.get("title"),
                    "text": item.get("text"),
                    "status": item.get("status"),
                    "error_message": item.get("errorMessage"),
                }
                for item in raw_lyrics
            ]

    raw_lyrics = container.get("data")
    if isinstance(raw_lyrics, list) and raw_lyrics and isinstance(raw_lyrics[0], dict) and "text" in raw_lyrics[0]:
        return [
            {
                "title": item.get("title"),
                "text": item.get("text"),
                "status": item.get("status"),
                "error_message": item.get("errorMessage"),
            }
            for item in raw_lyrics
        ]

    return []


def is_lyrics_success(status: Optional[str], lyrics: List[Dict[str, Any]]) -> bool:
    if (status or "").upper() in LYRICS_SUCCESS_STATUSES:
        return True
    return bool(lyrics) and any(item.get("text") for item in lyrics)


def is_lyrics_failure(status: Optional[str]) -> bool:
    return (status or "").upper() in LYRICS_FAILURE_STATUSES


def is_processing_success(flag: Optional[str], result_url: Optional[str] = None) -> bool:
    if (flag or "").upper() in PROCESSING_SUCCESS_FLAGS:
        return True
    return bool(result_url)


def is_processing_failure(flag: Optional[str]) -> bool:
    return (flag or "").upper() in PROCESSING_FAILURE_FLAGS
