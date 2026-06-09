"""Tests for P1 MCP tool registration and client methods."""

import inspect

from mcp_p1 import get_p1_tools
from suno_client import SunoClient


def test_p1_tools_registered():
    names = {tool.name for tool in get_p1_tools()}
    expected = {
        "generate_lyrics",
        "get_lyrics_status",
        "extend_music",
        "separate_vocals",
        "get_vocal_separation_status",
        "create_music_video",
        "get_music_video_status",
        "generate_persona",
        "upload_and_cover_audio",
    }
    assert expected.issubset(names)


def test_p1_client_methods_exist():
    for method in [
        "generate_lyrics",
        "get_lyrics_status",
        "extend_music",
        "separate_vocals",
        "get_vocal_separation_status",
        "create_music_video",
        "get_music_video_status",
        "generate_persona",
        "upload_and_cover_audio",
    ]:
        assert hasattr(SunoClient, method)
        assert inspect.iscoroutinefunction(getattr(SunoClient, method))


if __name__ == "__main__":
    test_p1_tools_registered()
    test_p1_client_methods_exist()
    print("P1 tests passed")
