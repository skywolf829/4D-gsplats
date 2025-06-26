import tempfile
import subprocess
import librosa
import numpy as np
import scipy.signal
from pathlib import Path
from file_ops import get_video_resolution, get_rotation_metadata, FFMPEG_FLAGS

# ---- FFmpeg-based utilities ----

def video_has_audio(video_path: Path) -> bool:
    """Returns True if the video has at least one audio stream."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0", str(video_path)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return bool(result.stdout.strip())

def extract_audio(video_path: Path, sr: int = 16000) -> Path:
    """Extracts mono audio from a video file to a temp WAV file."""
    tmp_wav = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    cmd = [
        "ffmpeg", *FFMPEG_FLAGS,
        "-i", str(video_path),
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        str(tmp_wav)
    ]
    subprocess.run(cmd, check=True)
    return tmp_wav

def compute_offset(ref_audio: Path, other_audio: Path, sr: int = 16000) -> float:
    """Returns offset in seconds (float) between other_audio and ref_audio."""
    y_ref, _ = librosa.load(str(ref_audio), sr=sr)
    y_other, _ = librosa.load(str(other_audio), sr=sr)

    corr = scipy.signal.correlate(y_ref, y_other, mode="full")
    lag = np.argmax(corr) - len(y_other)

    return float(lag) / sr  # in seconds

def sync_videos_get_offset(video_paths: list[Path], sr: int = 16000) -> list[float]:
    """
    Given N video paths, return a dict of video_path -> offset_ms
    relative to the first video.
    """
    if len(video_paths) < 2:
        raise ValueError(f"Need at least two videos for synchronization, got {video_paths}.")
    for path in video_paths:
        if not video_has_audio(path):
            raise ValueError(f"Video '{path}' does not contain an audio stream.")

    print("Extracting audio...")
    audio_paths = [extract_audio(v, sr=sr) for v in video_paths]

    ref_audio = audio_paths[0]
    offsets: list[float] = [0]

    print("Computing synchronization offsets...")
    for video, audio in zip(video_paths[1:], audio_paths[1:]):
        offset_ms = compute_offset(ref_audio, audio, sr=sr)
        offsets.append(offset_ms)

    for a in audio_paths:
        a.unlink()

    return offsets

def compute_synced_time_range(offsets: list[float], durations: list[float]) -> tuple[float, float]:
    """
    Given offsets (in seconds) and durations (in seconds), return the start and end
    time range (in seconds) over which all videos are valid after syncing.
    """
    assert len(offsets) == len(durations), f"Length of offsets and durations should be the same, got {len(offsets)} and {len(durations)}"
    # Start of valid content is offset
    # End of valid content is offset + duration
    start_times = offsets
    end_times = [offsets[k] + durations[k] for k in range(len(offsets))]

    common_start = max(start_times)
    common_end = min(end_times)

    if common_end <= common_start:
        raise ValueError("No overlapping synced time range exists.")

    return common_start, common_end

