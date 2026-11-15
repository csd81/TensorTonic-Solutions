import math


def video_input_layout(
    duration_seconds: int,
    max_frames: int,
    mode: str,
    tokens_per_frame: int,
    include_audio: bool,
) -> tuple[list[int], list[tuple], int]:
    """
    Returns the kept frame indices, the token layout and the soft token count.
    """
    def format_mmss(sec: int) -> str:
        return f"{sec // 60:02d}:{sec % 60:02d}"

    # 1. Select kept frame indices
    if duration_seconds <= max_frames:
        kept_frames = list(range(duration_seconds))
    elif mode == "first":
        kept_frames = list(range(max_frames))
    elif mode == "uniform":
        kept_frames = [
            math.floor((j * (duration_seconds - 1)) / (max_frames - 1) + 0.5)
            for j in range(max_frames)
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 2. Build interleaved sequence layout and count soft tokens
    layout: list[tuple] = [("BOS", 1)]
    soft_token_count = 0

    for idx, frame_sec in enumerate(kept_frames):
        # Image segment
        layout.extend([
            ("timestamp", format_mmss(frame_sec)),
            ("SOI", 1),
            ("image", tokens_per_frame),
            ("EOI", 1),
        ])
        soft_token_count += tokens_per_frame

        # Audio segments covering [frame_sec, next_frame_sec)
        if include_audio:
            next_sec = (
                kept_frames[idx + 1]
                if idx + 1 < len(kept_frames)
                else duration_seconds
            )
            for audio_sec in range(frame_sec, next_sec):
                layout.extend([
                    ("timestamp", format_mmss(audio_sec)),
                    ("SOA", 1),
                    ("audio", 25),
                    ("EOA", 1),
                ])
                soft_token_count += 25

    layout.append(("EOS", 1))

    return kept_frames, layout, soft_token_count
