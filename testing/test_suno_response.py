#!/usr/bin/env python3
"""Unit tests for Suno response normalization."""

from suno_response import (
    extract_generation_task_id,
    extract_tracks_from_container,
    is_terminal_failure,
    is_terminal_success,
    normalize_track,
)


def test_normalize_track_supports_camel_case():
    track = normalize_track(
        {
            "id": "abc",
            "title": "Song",
            "audioUrl": "https://example.com/a.mp3",
            "modelName": "chirp-v5",
            "createTime": "2025-01-01",
        }
    )
    assert track["audio_url"] == "https://example.com/a.mp3"
    assert track["model_name"] == "chirp-v5"


def test_extract_tracks_from_suno_data():
    data = {
        "status": "SUCCESS",
        "response": {
            "sunoData": [{"id": "1", "audioUrl": "https://example.com/1.mp3"}]
        },
    }
    tracks = extract_tracks_from_container(data)
    assert len(tracks) == 1
    assert tracks[0]["audio_url"] == "https://example.com/1.mp3"


def test_extract_tracks_from_response_data():
    data = {
        "status": "SUCCESS",
        "response": {
            "data": [{"id": "2", "audio_url": "https://example.com/2.mp3"}]
        },
    }
    tracks = extract_tracks_from_container(data)
    assert len(tracks) == 1
    assert tracks[0]["id"] == "2"


def test_terminal_status_helpers():
    assert is_terminal_success("SUCCESS", [])
    assert is_terminal_success("GENERATING", [{"audio_url": "https://example.com/a.mp3"}])
    assert is_terminal_failure("FAILED")


def test_extract_generation_task_id():
    result = {"data": {"taskId": "task123"}}
    assert extract_generation_task_id(result) == "task123"
