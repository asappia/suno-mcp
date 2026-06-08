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
