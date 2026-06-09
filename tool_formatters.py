"""Shared text formatters for MCP tool responses."""


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
