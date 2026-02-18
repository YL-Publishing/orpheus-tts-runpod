"""
RunPod Orpheus TTS Client — called from produce_recap.py on VPS.

Sends narration text to the RunPod Orpheus endpoint and receives WAV audio.
Supports both single and batch (full-chapter) modes.

Environment variables:
    RUNPOD_API_KEY       - RunPod API key (required)
    ORPHEUS_ENDPOINT_ID  - RunPod serverless endpoint ID (required)
    ORPHEUS_VOICE        - Default voice (optional, default: tara)

Usage:
    from orpheus_client import OrpheusClient

    client = OrpheusClient()

    # Single segment
    wav_bytes = client.generate("Hello world", voice="tara")

    # Batch (efficient — one cold start for entire chapter)
    segments = [
        {"id": "N1", "text": "First line", "voice": "leo"},
        {"id": "N2", "text": "Second line", "voice": "tara"},
    ]
    results = client.generate_batch(segments)
    for seg_id, wav_bytes in results.items():
        with open(f"{seg_id}.wav", "wb") as f:
            f.write(wav_bytes)
"""

import os
import time
import base64
import requests
from pathlib import Path


class OrpheusClient:
    """Client for the RunPod Orpheus TTS serverless endpoint."""

    # Voice mapping for different character archetypes
    VOICE_MAP = {
        "narrator": "tara",
        "narrator_male": "leo",
        "young_male": "dan",
        "young_female": "jess",
        "mature_male": "zac",
        "mature_female": "leah",
        "child": "mia",
        "energetic": "zoe",
    }

    VALID_VOICES = {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}

    def __init__(self, api_key=None, endpoint_id=None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.endpoint_id = endpoint_id or os.environ.get("ORPHEUS_ENDPOINT_ID", "")
        self.default_voice = os.environ.get("ORPHEUS_VOICE", "tara")

        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not set")
        if not self.endpoint_id:
            raise ValueError("ORPHEUS_ENDPOINT_ID not set")

        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _resolve_voice(self, voice: str) -> str:
        """Resolve voice name (supports both direct names and archetype mapping)."""
        if voice in self.VALID_VOICES:
            return voice
        return self.VOICE_MAP.get(voice, self.default_voice)

    def generate(self, text: str, voice: str = "tara", timeout: int = 120) -> bytes:
        """Generate WAV audio for a single text segment.

        Args:
            text: Text to speak (can include Orpheus emotion tags like <laugh>)
            voice: Voice name or archetype key
            timeout: Request timeout in seconds

        Returns:
            WAV audio bytes
        """
        voice = self._resolve_voice(voice)

        payload = {
            "input": {
                "text": text,
                "voice": voice,
            }
        }

        resp = requests.post(
            f"{self.base_url}/runsync",
            headers=self.headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") == "COMPLETED":
            output = result["output"]
            if "error" in output:
                raise RuntimeError(f"Orpheus error: {output['error']}")
            return base64.b64decode(output["audio_base64"])

        elif result.get("status") == "FAILED":
            raise RuntimeError(f"RunPod job failed: {result.get('error', 'unknown')}")
        else:
            raise RuntimeError(f"Unexpected status: {result.get('status')}")

    def generate_batch(
        self, segments: list, timeout: int = 300
    ) -> dict:
        """Generate WAV audio for multiple segments in one request.

        Efficient for chapter rendering — single cold start for all segments.

        Args:
            segments: List of {"id": "N1", "text": "...", "voice": "tara"}
            timeout: Request timeout (batch takes longer)

        Returns:
            Dict mapping segment ID to WAV bytes: {"N1": b"...", "N2": b"..."}
        """
        # Resolve voices
        resolved = []
        for seg in segments:
            resolved.append({
                "id": seg["id"],
                "text": seg["text"],
                "voice": self._resolve_voice(seg.get("voice", self.default_voice)),
            })

        payload = {
            "input": {
                "segments": resolved,
            }
        }

        resp = requests.post(
            f"{self.base_url}/runsync",
            headers=self.headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") != "COMPLETED":
            raise RuntimeError(
                f"RunPod batch failed: {result.get('status')} - {result.get('error', '')}"
            )

        output = result["output"]
        audio_map = {}
        for item in output.get("results", []):
            seg_id = item["id"]
            if "error" in item:
                print(f"  WARNING: Segment {seg_id} failed: {item['error']}")
                continue
            audio_map[seg_id] = base64.b64decode(item["audio_base64"])

        return audio_map

    def generate_to_file(self, text: str, output_path, voice: str = "tara") -> Path:
        """Generate and save WAV to file."""
        output_path = Path(output_path)
        wav_bytes = self.generate(text, voice=voice)
        output_path.write_bytes(wav_bytes)
        return output_path

    def generate_batch_to_files(
        self, segments: list, output_dir, filename_pattern: str = "{id}.wav"
    ) -> dict:
        """Generate batch and save each segment to a file.

        Args:
            segments: List of {"id": "N1", "text": "...", "voice": "tara"}
            output_dir: Directory to save WAV files
            filename_pattern: Pattern with {id} placeholder

        Returns:
            Dict mapping segment ID to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_map = self.generate_batch(segments)
        file_map = {}

        for seg_id, wav_bytes in audio_map.items():
            filename = filename_pattern.format(id=seg_id)
            filepath = output_dir / filename
            filepath.write_bytes(wav_bytes)
            file_map[seg_id] = filepath

        return file_map


# ── Convenience function for produce_recap.py integration ──────────

_client = None


def get_client():
    """Get or create the singleton OrpheusClient."""
    global _client
    if _client is None:
        _client = OrpheusClient()
    return _client


def orpheus_tts(text: str, output_path: str, voice: str = "tara"):
    """Drop-in replacement for edge-tts in produce_recap.py.

    Args:
        text: Text to speak
        output_path: Where to save the WAV file
        voice: Orpheus voice name
    """
    client = get_client()
    client.generate_to_file(text, output_path, voice=voice)
