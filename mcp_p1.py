"""P1 MCP tool definitions and handlers."""

from __future__ import annotations

from typing import Any, Optional

from mcp.types import TextContent, Tool

from callback_server import resolve_callback_url
from suno_client import SunoClient
from suno_response import extract_generation_task_id, extract_lyrics_from_container
from tool_formatters import format_tracks_response


def _callback_line(callback_url: Optional[str], active_callback_url: Optional[str]) -> str:
    used = callback_url or active_callback_url or resolve_callback_url()
    return f"Callback URL: {used}\n\n"


def format_lyrics_response(lyrics: list[dict]) -> str:
    if not lyrics:
        return "No lyrics available yet.\n"
    parts = [f"Generated {len(lyrics)} lyrics variation(s):\n"]
    for index, item in enumerate(lyrics, 1):
        parts.append(f"Variation {index}:")
        parts.append(f"  Title: {item.get('title', 'N/A')}")
        parts.append(f"  Status: {item.get('status', 'N/A')}")
        if item.get("text"):
            parts.append(f"  Lyrics:\n{item['text']}")
        parts.append("")
    return "\n".join(parts)


def format_stem_urls(response_data: dict) -> str:
    if not response_data:
        return ""
    labels = [
        ("originUrl", "Original"),
        ("instrumentalUrl", "Instrumental"),
        ("vocalUrl", "Vocals"),
        ("backingVocalsUrl", "Backing Vocals"),
        ("drumsUrl", "Drums"),
        ("bassUrl", "Bass"),
        ("guitarUrl", "Guitar"),
        ("keyboardUrl", "Keyboard"),
        ("percussionUrl", "Percussion"),
        ("stringsUrl", "Strings"),
        ("synthUrl", "Synth"),
        ("fxUrl", "FX"),
        ("brassUrl", "Brass"),
        ("woodwindsUrl", "Woodwinds"),
    ]
    lines = []
    for key, label in labels:
        url = response_data.get(key)
        if url:
            lines.append(f"  {label}: {url}")
    return "\n".join(lines)


def get_p1_tools() -> list[Tool]:
    model_enum = ["v4", "v4.5", "v4.5plus", "v4.5all", "v5", "v5.5"]
    return [
        Tool(
            name="generate_lyrics",
            description="Generate AI lyrics from a prompt without creating audio. Returns multiple lyric variations. Generated files are retained for 15 days.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Lyrics theme/description (max 200 chars)"},
                    "wait_completion": {"type": "boolean", "default": True},
                    "callback_url": {"type": "string"},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="get_lyrics_status",
            description="Get lyrics generation task status and lyric variations.",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        ),
        Tool(
            name="extend_music",
            description="Extend an existing generated track. Model must match the source track. Generated files are retained for 15 days.",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio_id": {"type": "string", "description": "Track UUID to extend"},
                    "model_version": {"type": "string", "enum": model_enum, "default": "v5"},
                    "default_param_flag": {"type": "boolean", "default": True},
                    "prompt": {"type": "string"},
                    "style": {"type": "string"},
                    "title": {"type": "string"},
                    "continue_at": {"type": "number", "description": "Seconds into track to continue from"},
                    "wait_audio": {"type": "boolean", "default": True},
                    "callback_url": {"type": "string"},
                    "persona_id": {"type": "string"},
                    "persona_model": {"type": "string", "enum": ["style_persona", "voice_persona"]},
                    "negative_tags": {"type": "string"},
                    "vocal_gender": {"type": "string", "enum": ["m", "f"]},
                    "style_weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "weirdness_constraint": {"type": "number", "minimum": 0, "maximum": 1},
                    "audio_weight": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["audio_id"],
            },
        ),
        Tool(
            name="separate_vocals",
            description="Separate vocals from a generated track. separate_vocal (~10 credits) returns vocals+instrumental; split_stem (~50 credits) returns up to 12 stems.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Original generation task ID"},
                    "audio_id": {"type": "string", "description": "Track UUID"},
                    "separation_type": {
                        "type": "string",
                        "enum": ["separate_vocal", "split_stem"],
                        "default": "separate_vocal",
                    },
                    "callback_url": {"type": "string"},
                },
                "required": ["task_id", "audio_id"],
            },
        ),
        Tool(
            name="get_vocal_separation_status",
            description="Get vocal separation task status and stem download URLs.",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "string", "description": "Separation task ID"}},
                "required": ["task_id"],
            },
        ),
        Tool(
            name="create_music_video",
            description="Create an MP4 visualization video for a generated track. Requires generation task_id and track audio_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "audio_id": {"type": "string"},
                    "author": {"type": "string", "description": "Artist name shown in video (max 50 chars)"},
                    "domain_name": {"type": "string", "description": "Watermark domain (max 50 chars)"},
                    "callback_url": {"type": "string"},
                },
                "required": ["task_id", "audio_id"],
            },
        ),
        Tool(
            name="get_music_video_status",
            description="Get music video generation status and MP4 download URL.",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "string", "description": "Video task ID"}},
                "required": ["task_id"],
            },
        ),
        Tool(
            name="generate_persona",
            description="Create a reusable Persona from a completed track. Returns persona_id for use in generate_music, extend_music, or upload_and_cover_audio.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "audio_id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "style": {"type": "string"},
                    "vocal_start": {"type": "number", "description": "Analysis start seconds (segment 10-30s)"},
                    "vocal_end": {"type": "number", "description": "Analysis end seconds"},
                },
                "required": ["task_id", "audio_id", "name", "description"],
            },
        ),
        Tool(
            name="upload_and_cover_audio",
            description="Upload audio from a public URL and restyle it while keeping the melody. upload_url audio max 8 minutes (1 minute for v4.5all).",
            inputSchema={
                "type": "object",
                "properties": {
                    "upload_url": {"type": "string", "description": "Public URL of source audio"},
                    "prompt": {"type": "string"},
                    "make_instrumental": {"type": "boolean", "default": False},
                    "model_version": {"type": "string", "enum": model_enum, "default": "v5"},
                    "custom_mode": {"type": "boolean", "default": False},
                    "style": {"type": "string"},
                    "title": {"type": "string"},
                    "wait_audio": {"type": "boolean", "default": True},
                    "callback_url": {"type": "string"},
                    "persona_id": {"type": "string"},
                    "persona_model": {"type": "string", "enum": ["style_persona", "voice_persona"]},
                    "negative_tags": {"type": "string"},
                    "vocal_gender": {"type": "string", "enum": ["m", "f"]},
                    "style_weight": {"type": "number", "minimum": 0, "maximum": 1},
                    "weirdness_constraint": {"type": "number", "minimum": 0, "maximum": 1},
                    "audio_weight": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["upload_url"],
            },
        ),
    ]


async def handle_p1_tool(
    name: str,
    arguments: dict,
    suno_client: SunoClient,
    callback_store: Any,
    active_callback_url: Optional[str],
) -> Optional[list[TextContent]]:
    if name == "generate_lyrics":
        result = await suno_client.generate_lyrics(
            prompt=arguments["prompt"],
            callback_url=arguments.get("callback_url"),
            wait_completion=arguments.get("wait_completion", True),
            callback_store=callback_store,
        )
        text = "Lyrics generation completed!\n\n" if arguments.get("wait_completion", True) else "Lyrics generation started!\n\n"
        text += _callback_line(arguments.get("callback_url"), active_callback_url)
        data = result.get("data", {})
        if data.get("taskId"):
            text += f"Task ID: {data['taskId']}\n"
        lyrics = data.get("lyrics") or extract_lyrics_from_container(data)
        text += "\n" + format_lyrics_response(lyrics)
        return [TextContent(type="text", text=text)]

    if name == "get_lyrics_status":
        result = await suno_client.get_lyrics_status(arguments["task_id"])
        data = result.get("data", {})
        text = "Lyrics Task Status:\n\n"
        text += f"Task ID: {data.get('taskId', 'N/A')}\n"
        text += f"Status: {data.get('status', 'N/A')}\n\n"
        text += format_lyrics_response(extract_lyrics_from_container(data))
        return [TextContent(type="text", text=text)]

    if name == "extend_music":
        result = await suno_client.extend_music(
            audio_id=arguments["audio_id"],
            model_version=SunoClient.map_model_version(arguments.get("model_version", "v5")),
            default_param_flag=arguments.get("default_param_flag", True),
            prompt=arguments.get("prompt"),
            style=arguments.get("style"),
            title=arguments.get("title"),
            continue_at=arguments.get("continue_at"),
            callback_url=arguments.get("callback_url"),
            persona_id=arguments.get("persona_id"),
            persona_model=arguments.get("persona_model"),
            negative_tags=arguments.get("negative_tags"),
            vocal_gender=arguments.get("vocal_gender"),
            style_weight=arguments.get("style_weight"),
            weirdness_constraint=arguments.get("weirdness_constraint"),
            audio_weight=arguments.get("audio_weight"),
            wait_audio=arguments.get("wait_audio", True),
            callback_store=callback_store,
        )
        wait_audio = arguments.get("wait_audio", True)
        text = f"Music extension {'completed' if wait_audio else 'started'}!\n\n"
        text += _callback_line(arguments.get("callback_url"), active_callback_url)
        data = result.get("data", {})
        if data.get("taskId"):
            text += f"Task ID: {data['taskId']}\n"
        tracks = data.get("tracks")
        if tracks:
            text += "\n" + format_tracks_response(tracks)
        elif not wait_audio:
            text += "\nUse get_task_status with the task ID to check progress.\n"
        return [TextContent(type="text", text=text)]

    if name == "separate_vocals":
        result = await suno_client.separate_vocals(
            task_id=arguments["task_id"],
            audio_id=arguments["audio_id"],
            separation_type=arguments.get("separation_type", "separate_vocal"),
            callback_url=arguments.get("callback_url"),
        )
        text = "Vocal separation started!\n\n"
        text += _callback_line(arguments.get("callback_url"), active_callback_url)
        task_id = extract_generation_task_id(result)
        if task_id:
            text += f"Separation Task ID: {task_id}\n"
            text += "Use get_vocal_separation_status with this task ID to retrieve stem URLs.\n"
        return [TextContent(type="text", text=text)]

    if name == "get_vocal_separation_status":
        result = await suno_client.get_vocal_separation_status(arguments["task_id"])
        data = result.get("data", {})
        text = "Vocal Separation Status:\n\n"
        text += f"Task ID: {data.get('taskId', 'N/A')}\n"
        text += f"Status: {data.get('successFlag', 'N/A')}\n"
        stems = format_stem_urls(data.get("response", {}))
        if stems:
            text += "\nStem URLs:\n" + stems + "\n"
        elif data.get("successFlag") != "SUCCESS":
            text += "\nSeparation still in progress.\n"
        return [TextContent(type="text", text=text)]

    if name == "create_music_video":
        result = await suno_client.create_music_video(
            task_id=arguments["task_id"],
            audio_id=arguments["audio_id"],
            callback_url=arguments.get("callback_url"),
            author=arguments.get("author"),
            domain_name=arguments.get("domain_name"),
        )
        text = "Music video generation started!\n\n"
        text += _callback_line(arguments.get("callback_url"), active_callback_url)
        task_id = extract_generation_task_id(result)
        if task_id:
            text += f"Video Task ID: {task_id}\n"
            text += "Use get_music_video_status with this task ID to retrieve the MP4 URL.\n"
        return [TextContent(type="text", text=text)]

    if name == "get_music_video_status":
        result = await suno_client.get_music_video_status(arguments["task_id"])
        data = result.get("data", {})
        text = "Music Video Status:\n\n"
        text += f"Task ID: {data.get('taskId', 'N/A')}\n"
        text += f"Status: {data.get('successFlag', 'N/A')}\n"
        video_url = (data.get("response") or {}).get("videoUrl")
        if video_url:
            text += f"\nVideo URL: {video_url}\n"
        else:
            text += "\nVideo still processing.\n"
        return [TextContent(type="text", text=text)]

    if name == "generate_persona":
        result = await suno_client.generate_persona(
            task_id=arguments["task_id"],
            audio_id=arguments["audio_id"],
            name=arguments["name"],
            description=arguments["description"],
            style=arguments.get("style"),
            vocal_start=arguments.get("vocal_start"),
            vocal_end=arguments.get("vocal_end"),
        )
        data = result.get("data", {})
        text = "Persona created!\n\n"
        text += f"Persona ID: {data.get('personaId', 'N/A')}\n"
        text += f"Name: {data.get('name', arguments['name'])}\n"
        text += f"Description: {data.get('description', arguments['description'])}\n"
        text += "\nUse persona_id in generate_music, extend_music, or upload_and_cover_audio.\n"
        return [TextContent(type="text", text=text)]

    if name == "upload_and_cover_audio":
        result = await suno_client.upload_and_cover_audio(
            upload_url=arguments["upload_url"],
            model_version=SunoClient.map_model_version(arguments.get("model_version", "v5")),
            custom_mode=arguments.get("custom_mode", False),
            make_instrumental=arguments.get("make_instrumental", False),
            prompt=arguments.get("prompt"),
            style=arguments.get("style"),
            title=arguments.get("title"),
            callback_url=arguments.get("callback_url"),
            persona_id=arguments.get("persona_id"),
            persona_model=arguments.get("persona_model"),
            negative_tags=arguments.get("negative_tags"),
            vocal_gender=arguments.get("vocal_gender"),
            style_weight=arguments.get("style_weight"),
            weirdness_constraint=arguments.get("weirdness_constraint"),
            audio_weight=arguments.get("audio_weight"),
            wait_audio=arguments.get("wait_audio", True),
            callback_store=callback_store,
        )
        wait_audio = arguments.get("wait_audio", True)
        text = f"Upload-cover {'completed' if wait_audio else 'started'}!\n\n"
        text += _callback_line(arguments.get("callback_url"), active_callback_url)
        data = result.get("data", {})
        if data.get("taskId"):
            text += f"Task ID: {data['taskId']}\n"
        tracks = data.get("tracks")
        if tracks:
            text += "\n" + format_tracks_response(tracks)
        elif not wait_audio:
            text += "\nUse get_task_status with the task ID to check progress.\n"
        return [TextContent(type="text", text=text)]

    return None
