#!/usr/bin/env python3
"""Suno MCP Server - AI Music Generation via Model Context Protocol."""

import asyncio
import os
from typing import Any, Optional

from dotenv import load_dotenv

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from callback_server import resolve_callback_url, start_callback_server
from callback_store import CallbackStore
from suno_client import SunoClient, SunoAPIError
from suno_response import extract_generation_task_id, extract_tracks_from_container, normalize_track

# Load environment variables
load_dotenv()

# Initialize server
server = Server("suno-mcp-server")

# Global client and callback infrastructure
suno_client: SunoClient | None = None
callback_store = CallbackStore()
callback_runner: Any = None
active_callback_url: Optional[str] = None


def format_track_lines(track: dict, index: int) -> str:
    lines = [
        f"Track {index}:",
        f"  ID: {track.get('id', 'N/A')}",
        f"  Title: {track.get('title', 'N/A')}",
    ]

    if track.get("status"):
        lines.append(f"  Status: {track['status']}")
    if track.get("model_name"):
        lines.append(f"  Model: {track['model_name']}")
    if track.get("duration"):
        lines.append(f"  Duration: {track['duration']}s")
    if track.get("tags"):
        lines.append(f"  Tags: {track['tags']}")
    if track.get("audio_url"):
        lines.append(f"  Audio URL: {track['audio_url']}")
    if track.get("stream_audio_url"):
        lines.append(f"  Stream URL: {track['stream_audio_url']}")
    if track.get("video_url"):
        lines.append(f"  Video URL: {track['video_url']}")
    if track.get("image_url"):
        lines.append(f"  Image URL: {track['image_url']}")
    if track.get("create_time") or track.get("created_at"):
        lines.append(f"  Created: {track.get('create_time') or track.get('created_at')}")

    return "\n".join(lines)


def format_tracks_response(tracks: list[dict]) -> str:
    if not tracks:
        return "No track details available yet.\n"

    parts = [f"Generated {len(tracks)} track(s):\n"]
    for index, track in enumerate(tracks, 1):
        parts.append(format_track_lines(track, index))
        parts.append("")
    return "\n".join(parts)


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available Suno API tools."""
    return [
        Tool(
            name="generate_music",
            description=(
                "Generate AI music from a text prompt. Creates high-quality music in various styles and genres. "
                "Supports both simple and custom modes with advanced controls. "
                "If callback_url is omitted, the server uses its built-in webhook receiver. "
                "For Docker, publish port 8090 and set SUNO_CALLBACK_PUBLIC_URL to a public URL (e.g. ngrok) "
                "so Suno can reach the callback endpoint."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description/lyrics for the music. In Custom Mode with make_instrumental=false, this is used as exact lyrics (max 3000-5000 chars). In Non-custom Mode, used as core idea for auto-generated lyrics (max 500 chars). Not required if custom_mode=true and make_instrumental=true."
                    },
                    "make_instrumental": {
                        "type": "boolean",
                        "description": "If true, generate instrumental only without vocals",
                        "default": False
                    },
                    "model_version": {
                        "type": "string",
                        "description": "AI model version to use. V5 and V5_5 offer the best quality and speed.",
                        "enum": ["v4", "v4.5", "v4.5plus", "v4.5all", "v5", "v5.5"],
                        "default": "v5"
                    },
                    "custom_mode": {
                        "type": "boolean",
                        "description": "Enable custom mode for advanced control. When true, requires style and title parameters. Allows you to specify exact lyrics, music style, and other advanced settings.",
                        "default": False
                    },
                    "style": {
                        "type": "string",
                        "description": "Music style/genre (required in Custom Mode, e.g., 'orchestral epic, cinematic, powerful strings'). Max 200-1000 chars depending on model."
                    },
                    "title": {
                        "type": "string",
                        "description": "Song title (required in Custom Mode). Max 80-100 characters depending on model."
                    },
                    "wait_audio": {
                        "type": "boolean",
                        "description": "If true, poll task status until generation completes before returning",
                        "default": True
                    },
                    "callback_url": {
                        "type": "string",
                        "description": "Optional webhook URL. If omitted, the MCP server uses its built-in callback endpoint automatically."
                    },
                    "persona_id": {
                        "type": "string",
                        "description": "Persona identifier for stylistic influence (Custom Mode only)"
                    },
                    "persona_model": {
                        "type": "string",
                        "description": "Persona type when using persona_id",
                        "enum": ["style_persona", "voice_persona"]
                    },
                    "negative_tags": {
                        "type": "string",
                        "description": "Styles or traits to exclude from generation (e.g., 'aggressive, heavy metal')"
                    },
                    "vocal_gender": {
                        "type": "string",
                        "description": "Preferred vocal gender",
                        "enum": ["m", "f"]
                    },
                    "style_weight": {
                        "type": "number",
                        "description": "Weight of style guidance (0.00-1.00). Higher values adhere more strictly to the specified style.",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "weirdness_constraint": {
                        "type": "number",
                        "description": "Creative deviation tolerance (0.00-1.00). Higher values allow more experimental/unusual results.",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "audio_weight": {
                        "type": "number",
                        "description": "Input audio influence weighting (0.00-1.00)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_task_status",
            description="Get the status of a music generation task using the taskId returned from generate_music. This shows generation progress and track information once complete.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID returned from generate_music"
                    }
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="get_music_info",
            description="Get detailed information about generated music tracks using track IDs (not task IDs). Use this for tracks you already have the specific track IDs for.",
            inputSchema={
                "type": "object",
                "properties": {
                    "track_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of track IDs to retrieve information for"
                    }
                },
                "required": ["track_ids"]
            }
        ),
        Tool(
            name="get_credits",
            description="Check your Suno API account credit balance and usage statistics.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="convert_to_wav",
            description="Convert a generated MP3 track to high-quality WAV format. Requires BOTH task_id (generation job ID) AND audio_id (specific track ID). CRITICAL FOR AI: Returns a NEW conversion task_id (different from generation task_id) which you MUST save and provide to the user. This conversion task_id is the ONLY way to retrieve the WAV download URL later. The API does NOT support querying by audio_id alone. If the conversion task_id is lost, the WAV URL is permanently unrecoverable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "callback_url": {
                        "type": "string",
                        "description": "Optional webhook URL. If omitted, the MCP server uses its built-in callback endpoint automatically."
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Generation task ID (taskId) from music['data']['taskId'] - the hex string identifying the generation job (REQUIRED)"
                    },
                    "audio_id": {
                        "type": "string",
                        "description": "Track ID (audioId) from track results - UUID identifying the specific track to convert (REQUIRED)"
                    }
                },
                "required": ["task_id", "audio_id"]
            }
        ),
        Tool(
            name="get_wav_conversion_status",
            description="Get the status of a WAV conversion task and retrieve the WAV download URL. CRITICAL: You must use the CONVERSION task_id returned from convert_to_wav (NOT the generation task_id from generate_music). These are two different task IDs. The conversion task_id is the ONLY way to retrieve the WAV download URL - there is no other method to query by audio_id or generation task_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID returned from convert_to_wav"
                    }
                },
                "required": ["task_id"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool execution requests."""
    if not suno_client:
        raise RuntimeError("Suno client not initialized")

    arguments = arguments or {}

    try:
        if name == "generate_music":
            prompt = arguments.get("prompt")
            make_instrumental = arguments.get("make_instrumental", False)
            model_version = arguments.get("model_version", "v5")
            api_model_version = SunoClient.map_model_version(model_version)

            custom_mode = arguments.get("custom_mode", False)
            style = arguments.get("style")
            title = arguments.get("title")
            wait_audio = arguments.get("wait_audio", True)
            callback_url = arguments.get("callback_url")
            persona_id = arguments.get("persona_id")
            persona_model = arguments.get("persona_model")
            negative_tags = arguments.get("negative_tags")
            vocal_gender = arguments.get("vocal_gender")
            style_weight = arguments.get("style_weight")
            weirdness_constraint = arguments.get("weirdness_constraint")
            audio_weight = arguments.get("audio_weight")

            result = await suno_client.generate_music(
                prompt=prompt,
                make_instrumental=make_instrumental,
                model_version=api_model_version,
                wait_audio=wait_audio,
                custom_mode=custom_mode,
                style=style,
                title=title,
                callback_url=callback_url,
                persona_id=persona_id,
                persona_model=persona_model,
                negative_tags=negative_tags,
                vocal_gender=vocal_gender,
                style_weight=style_weight,
                weirdness_constraint=weirdness_constraint,
                audio_weight=audio_weight,
                callback_store=callback_store,
            )

            used_callback_url = callback_url or active_callback_url or resolve_callback_url()
            response_text = f"Music generation {'completed' if wait_audio else 'started'}!\n\n"
            response_text += f"Callback URL: {used_callback_url}\n\n"

            data = result.get("data")
            if isinstance(data, dict):
                task_id = data.get("taskId") or extract_generation_task_id(result)
                if task_id:
                    response_text += f"Task ID: {task_id}\n"

                tracks = data.get("tracks")
                if tracks:
                    response_text += "\n" + format_tracks_response(tracks)
                elif not wait_audio:
                    response_text += (
                        "\nNote: Generation is processing asynchronously. "
                        "Use get_task_status with the task ID to check progress.\n"
                    )
                elif wait_audio:
                    response_text += "\nGeneration finished but no track details were returned yet.\n"
            else:
                response_text += f"Response: {result}\n"

            return [TextContent(type="text", text=response_text)]

        elif name == "get_task_status":
            task_id = arguments.get("task_id")
            if not task_id:
                raise ValueError("task_id is required")

            result = await suno_client.get_task_status(task_id)
            response_text = "Music Generation Task Status:\n\n"

            if "data" in result:
                data = result["data"]
                response_text += f"Task ID: {data.get('taskId', 'N/A')}\n"
                response_text += f"Status: {data.get('status', 'N/A')}\n"
                response_text += f"Operation: {data.get('operationType', 'N/A')}\n"
                response_text += f"Model: {data.get('type', 'N/A')}\n"

                tracks = extract_tracks_from_container(data)
                if tracks:
                    response_text += "\n" + format_tracks_response(tracks)
                else:
                    response_text += f"\nGeneration in progress. Current status: {data.get('status', 'UNKNOWN')}\n"
            else:
                response_text += f"Response: {result}\n"

            return [TextContent(type="text", text=response_text)]

        elif name == "get_music_info":
            track_ids = arguments.get("track_ids")
            if not track_ids or not isinstance(track_ids, list):
                raise ValueError("track_ids must be a non-empty list")

            result = await suno_client.get_music_info(track_ids)
            response_text = "Music Track Information:\n\n"

            if "data" in result:
                raw_tracks = result["data"]
                if isinstance(raw_tracks, list):
                    tracks = [normalize_track(track) for track in raw_tracks]
                    response_text += format_tracks_response(tracks)
                    if tracks and tracks[0].get("prompt"):
                        response_text += f"Prompt: {tracks[0]['prompt']}\n"
                else:
                    response_text += f"Data: {raw_tracks}\n"
            else:
                response_text += f"Response: {result}\n"

            return [TextContent(type="text", text=response_text)]

        elif name == "get_credits":
            result = await suno_client.get_credits()
            response_text = "Suno API Credits:\n\n"

            if "data" in result:
                data = result["data"]
                if isinstance(data, (int, float)):
                    response_text += f"Remaining Credits: {data}\n"
                else:
                    response_text += f"Total Credits: {data.get('total_credits', 'N/A')}\n"
                    response_text += f"Used Credits: {data.get('used_credits', 'N/A')}\n"
                    response_text += f"Remaining Credits: {data.get('remaining_credits', 'N/A')}\n"
            else:
                response_text += f"Response: {result}\n"

            return [TextContent(type="text", text=response_text)]

        elif name == "convert_to_wav":
            callback_url = arguments.get("callback_url")
            task_id = arguments.get("task_id")
            audio_id = arguments.get("audio_id")

            if not task_id:
                raise ValueError("task_id is required (generation job ID from music['data']['taskId'])")

            if not audio_id:
                raise ValueError("audio_id is required (track ID from generated track results)")

            result = await suno_client.convert_to_wav(
                task_id=task_id,
                audio_id=audio_id,
                callback_url=callback_url or active_callback_url or resolve_callback_url(),
            )

            response_text = "WAV Conversion Started!\n\n"
            used_callback_url = callback_url or active_callback_url or resolve_callback_url()
            response_text += f"Callback URL: {used_callback_url}\n\n"

            if "data" in result and result["data"] is not None:
                data = result["data"]

                if isinstance(data, dict) and "taskId" in data:
                    conversion_task_id = data['taskId']
                    response_text += f"🎵 WAV Conversion Initiated\n\n"
                    response_text += f"=" * 60 + "\n"
                    response_text += f"⚠️  CRITICAL: SAVE THIS CONVERSION TASK ID  ⚠️\n"
                    response_text += f"=" * 60 + "\n\n"
                    response_text += f"Conversion Task ID: {conversion_task_id}\n\n"
                    response_text += f"This is a DIFFERENT ID from the generation task_id!\n"
                    response_text += f"Original Generation Task ID: {task_id}\n"
                    response_text += f"Track Audio ID: {audio_id}\n\n"
                    response_text += f"🔴 WITHOUT THIS CONVERSION TASK ID, YOU CANNOT:\n"
                    response_text += f"   - Retrieve the WAV download URL\n"
                    response_text += f"   - Check conversion status\n"
                    response_text += f"   - Access the converted file\n\n"
                    response_text += f"✅ TO GET THE WAV DOWNLOAD URL:\n"
                    response_text += f"   Use: get_wav_conversion_status(task_id=\"{conversion_task_id}\")\n\n"
                    response_text += f"⚠️  The Suno API has NO other way to retrieve the WAV URL.\n"
                    response_text += f"   You CANNOT query by audio_id or generation task_id.\n"
                    response_text += f"   If you lose this conversion task_id, the WAV is unrecoverable.\n"
                else:
                    response_text += f"Data: {data}\n"
            else:
                response_text += f"Response: {result}\n"

            return [TextContent(type="text", text=response_text)]

        elif name == "get_wav_conversion_status":
            task_id = arguments.get("task_id")
            if not task_id:
                raise ValueError("task_id is required")

            result = await suno_client.get_wav_conversion_status(task_id)
            response_text = "WAV Conversion Task Status:\n\n"

            if "data" in result and result["data"] is not None:
                data = result["data"]

                response_text += f"Task ID: {data.get('taskId', 'N/A')}\n"
                response_text += f"Music ID: {data.get('musicId', 'N/A')}\n"
                response_text += f"Status: {data.get('successFlag', 'N/A')}\n"

                response_data = data.get('response')
                if response_data and isinstance(response_data, dict):
                    wav_url = response_data.get('audioWavUrl')
                    if wav_url:
                        response_text += "\n✅ WAV Conversion Complete!\n"
                        response_text += f"WAV Download URL: {wav_url}\n"

                        if data.get('completeTime'):
                            response_text += f"Completed: {data['completeTime']}\n"
                        if data.get('createTime'):
                            response_text += f"Created: {data['createTime']}\n"
                    else:
                        response_text += f"\n⏳ Conversion in progress...\n"
                        response_text += f"Current Status: {data.get('successFlag', 'PENDING')}\n"
                else:
                    response_text += f"\n⏳ Conversion in progress...\n"
                    response_text += f"Current Status: {data.get('successFlag', 'PENDING')}\n"

                if data.get('errorCode') or data.get('errorMessage'):
                    response_text += f"\n⚠️ Error Details:\n"
                    response_text += f"Error Code: {data.get('errorCode', 'N/A')}\n"
                    response_text += f"Error Message: {data.get('errorMessage', 'N/A')}\n"
            else:
                response_text += f"Response: {result}\n"

            return [TextContent(type="text", text=response_text)]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except SunoAPIError as e:
        return [TextContent(type="text", text=f"Suno API Error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    global suno_client, callback_runner, active_callback_url

    try:
        suno_client = SunoClient()
    except Exception:
        return

    callback_runner, active_callback_url = await start_callback_server(callback_store)

    async with stdio_server() as (read_stream, write_stream):
        try:
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="suno-mcp-server",
                    server_version="1.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
        finally:
            if suno_client:
                await suno_client.close()
            if callback_runner:
                await callback_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
