from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path


TOKEN_ALIASES = {
    "techer": "teacher",
    "sudent": "student",
}


@dataclass
class ScriptedFrameSignal:
    frame_index: int
    signal_score: float
    motion_energy: float
    status: str
    left_hand_present: bool
    right_hand_present: bool
    pose_present: bool


@dataclass
class ScriptedSegment:
    start_frame: int
    end_frame: int
    mean_signal: float
    peak_signal: float
    mean_motion: float
    peak_motion: float

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    @property
    def strength(self) -> float:
        return (self.mean_signal * 0.65) + (self.peak_signal * 0.20) + (self.mean_motion * 0.15)


def parse_script_tokens_from_path(video_path: str | Path, allowed_labels: set[str] | None = None) -> list[str]:
    path = Path(video_path)
    stem = re.sub(r"\s+\(\d+\)$", "", path.stem)
    raw_tokens = [token.strip().lower() for token in stem.split("_") if token.strip()]
    normalized = [TOKEN_ALIASES.get(token, token) for token in raw_tokens]
    if allowed_labels is None:
        return normalized
    filtered = [token for token in normalized if token in allowed_labels]
    return filtered if filtered else []


def detect_activity_segments(
    frame_signals: list[ScriptedFrameSignal],
    *,
    signal_threshold: float = 0.20,
    gap_frames: int = 15,
    min_segment_frames: int = 10,
) -> list[ScriptedSegment]:
    blank_statuses = {"invalid_frame", "insufficient_signal", "collecting"}
    active_start: int | None = None
    blank_count = 0
    segments: list[ScriptedSegment] = []

    def finalize(end_index: int) -> None:
        nonlocal active_start
        if active_start is None or end_index < active_start:
            active_start = None
            return
        span = frame_signals[active_start : end_index + 1]
        if len(span) < min_segment_frames:
            active_start = None
            return
        mean_signal = sum(item.signal_score for item in span) / len(span)
        peak_signal = max(item.signal_score for item in span)
        mean_motion = sum(item.motion_energy for item in span) / len(span)
        peak_motion = max(item.motion_energy for item in span)
        segments.append(
            ScriptedSegment(
                start_frame=span[0].frame_index,
                end_frame=span[-1].frame_index,
                mean_signal=mean_signal,
                peak_signal=peak_signal,
                mean_motion=mean_motion,
                peak_motion=peak_motion,
            )
        )
        active_start = None

    for idx, item in enumerate(frame_signals):
        has_hand = item.left_hand_present or item.right_hand_present
        is_active = (
            has_hand
            and item.pose_present
            and item.signal_score >= signal_threshold
            and item.status not in blank_statuses
        )
        if is_active:
            if active_start is None:
                active_start = idx
            blank_count = 0
            continue
        if active_start is None:
            continue
        blank_count += 1
        if blank_count >= gap_frames:
            finalize(idx - gap_frames)
            blank_count = 0

    if active_start is not None:
        finalize(len(frame_signals) - 1)
    return segments


def _merge_weakest_adjacent(segments: list[ScriptedSegment]) -> list[ScriptedSegment]:
    if len(segments) <= 1:
        return segments
    weakest_index = 0
    weakest_score = float("inf")
    for idx in range(len(segments) - 1):
        left = segments[idx]
        right = segments[idx + 1]
        gap_penalty = max(0, right.start_frame - left.end_frame - 1) * 0.002
        score = left.strength + right.strength + gap_penalty
        if score < weakest_score:
            weakest_score = score
            weakest_index = idx
    left = segments[weakest_index]
    right = segments[weakest_index + 1]
    merged = ScriptedSegment(
        start_frame=left.start_frame,
        end_frame=right.end_frame,
        mean_signal=(left.mean_signal * left.frame_count + right.mean_signal * right.frame_count)
        / max(left.frame_count + right.frame_count, 1),
        peak_signal=max(left.peak_signal, right.peak_signal),
        mean_motion=(left.mean_motion * left.frame_count + right.mean_motion * right.frame_count)
        / max(left.frame_count + right.frame_count, 1),
        peak_motion=max(left.peak_motion, right.peak_motion),
    )
    return segments[:weakest_index] + [merged] + segments[weakest_index + 2 :]


def _split_longest_segment(segments: list[ScriptedSegment], frame_signals: list[ScriptedFrameSignal]) -> list[ScriptedSegment]:
    if not segments:
        return segments
    longest_index = max(range(len(segments)), key=lambda idx: segments[idx].frame_count)
    segment = segments[longest_index]
    if segment.frame_count < 2:
        return segments
    start = segment.start_frame
    end = segment.end_frame
    candidate_start = start + max(5, segment.frame_count // 4)
    candidate_end = end - max(5, segment.frame_count // 4)
    if candidate_end <= candidate_start:
        split_frame = start + segment.frame_count // 2
    else:
        slice_rows = frame_signals[candidate_start : candidate_end + 1]
        split_row = min(
            slice_rows,
            key=lambda item: (item.signal_score * 0.7) + (item.motion_energy * 0.3),
        )
        split_frame = split_row.frame_index
    left_span = frame_signals[start : split_frame + 1]
    right_span = frame_signals[split_frame + 1 : end + 1]
    if not left_span or not right_span:
        midpoint = start + segment.frame_count // 2
        left_span = frame_signals[start : midpoint + 1]
        right_span = frame_signals[midpoint + 1 : end + 1]
    if not left_span or not right_span:
        return segments

    def build(span: list[ScriptedFrameSignal]) -> ScriptedSegment:
        return ScriptedSegment(
            start_frame=span[0].frame_index,
            end_frame=span[-1].frame_index,
            mean_signal=sum(item.signal_score for item in span) / len(span),
            peak_signal=max(item.signal_score for item in span),
            mean_motion=sum(item.motion_energy for item in span) / len(span),
            peak_motion=max(item.motion_energy for item in span),
        )

    left_segment = build(left_span)
    right_segment = build(right_span)
    return segments[:longest_index] + [left_segment, right_segment] + segments[longest_index + 1 :]


def adjust_segment_count(
    segments: list[ScriptedSegment],
    *,
    target_count: int,
    frame_signals: list[ScriptedFrameSignal],
) -> list[ScriptedSegment]:
    adjusted = list(segments)
    while len(adjusted) > target_count and len(adjusted) > 1:
        adjusted = _merge_weakest_adjacent(adjusted)
    while len(adjusted) < target_count and adjusted:
        previous = len(adjusted)
        adjusted = _split_longest_segment(adjusted, frame_signals)
        if len(adjusted) == previous:
            break
    return adjusted


def build_scripted_alignment(
    video_path: str | Path,
    frame_signals: list[ScriptedFrameSignal],
    *,
    allowed_labels: set[str] | None = None,
    signal_threshold: float = 0.20,
    gap_frames: int = 15,
    min_segment_frames: int = 10,
) -> dict[str, object] | None:
    target_tokens = parse_script_tokens_from_path(video_path, allowed_labels=allowed_labels)
    if not target_tokens:
        return None
    detected_segments = detect_activity_segments(
        frame_signals,
        signal_threshold=signal_threshold,
        gap_frames=gap_frames,
        min_segment_frames=min_segment_frames,
    )
    aligned_segments = adjust_segment_count(
        detected_segments,
        target_count=len(target_tokens),
        frame_signals=frame_signals,
    )
    aligned_segments = aligned_segments[: len(target_tokens)]
    tokens = target_tokens[: len(aligned_segments)]
    return {
        "source_video_path": str(video_path),
        "expected_tokens": target_tokens,
        "detected_segment_count": len(detected_segments),
        "aligned_segment_count": len(aligned_segments),
        "tokens": tokens,
        "segments": [
            {
                "token": token,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "frame_count": segment.frame_count,
                "mean_signal": round(segment.mean_signal, 6),
                "peak_signal": round(segment.peak_signal, 6),
                "mean_motion": round(segment.mean_motion, 6),
                "peak_motion": round(segment.peak_motion, 6),
            }
            for token, segment in zip(tokens, aligned_segments)
        ],
        "pass": len(aligned_segments) == len(target_tokens),
    }
