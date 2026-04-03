from __future__ import annotations

"""
Current limitations: want/teacher/father remain training/data-domain residuals (WER ~0.375)

Direct webcam entrypoint for the current best-known trigger-based serving stack.

Default behavior:
- webcam input from camera index 0
- mirrored preview
- trigger-based continuous inference
- on-screen emitted token stream

Optional environment variables:
- SIGN_SIMULATION_MODE=1
  Run against the bundled sample video instead of a webcam.
- SIGN_HEADLESS=1
  Disable cv2.imshow and print a JSON summary at the end.
- SIGN_MIRROR_INPUT=1
  Mirror frames before inference. Enabled by default to match the best-known stack.
- SIGN_CAMERA_INDEX=0
  Select webcam device index.
- SIGN_MAX_FRAMES=300
  Stop after a fixed number of frames.
"""

import json
import os
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent
INTEGRATION_ROOT = PROJECT_ROOT / "integration_workspace"
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from src.app.sign_sentence_engine import MultibranchSequenceEngine  # noqa: E402


CHECKPOINT_PATH = INTEGRATION_ROOT / "artifacts_webcam9_relative_coord_v1" / "models" / "multibranch_baseline.pt"
GESTURE_PROFILE_PATH = PROJECT_ROOT / "metadata" / "webcam9_gesture_profiles.json"
DEFAULT_SAMPLE_VIDEO = PROJECT_ROOT / "i_you_mother_father_techer_sudent_want_like (1).mp4"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "realtime_webcam_runner_last_summary.json"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.35
TRIGGER_EMIT_CONFIDENCE_THRESHOLD = 0.60


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _find_simulation_video() -> Path | None:
    if DEFAULT_SAMPLE_VIDEO.exists():
        return DEFAULT_SAMPLE_VIDEO
    candidates = sorted(PROJECT_ROOT.glob("*.mp4"))
    return candidates[0] if candidates else None


def _build_engine() -> MultibranchSequenceEngine:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")
    if not GESTURE_PROFILE_PATH.exists():
        raise FileNotFoundError(f"Missing gesture profile: {GESTURE_PROFILE_PATH}")

    engine = MultibranchSequenceEngine(
        checkpoint_path=CHECKPOINT_PATH,
        sequence_length=SEQUENCE_LENGTH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        gesture_profile_path=GESTURE_PROFILE_PATH,
        mode="trigger_based",
    )

    # Locked best-known stack. Do not change without fresh regression evidence.
    engine.pose_local_anchor = "torso_center"
    engine.trigger_emit_confidence_threshold = TRIGGER_EMIT_CONFIDENCE_THRESHOLD

    engine.enable_mother_nose_z_calibration = True
    engine.mother_nose_z_offset = -1.5

    engine.enable_you_i_leftbody_calibration = True
    engine.you_i_leftbody_y_offset = -0.035

    engine.enable_you_like_pairwise_calibration = True
    engine.you_like_pairwise_delta = 0.98

    engine.enable_like_i_pairwise_calibration = True
    engine.like_i_pairwise_delta = 2.78

    engine.enable_want_anchor_fallback = True
    engine.want_anchor_fallback_max_nosign_confidence = 0.95

    engine.enable_father_trigger_rescue = True
    engine.father_trigger_rescue_pairwise_delta = 6.50
    engine.father_trigger_rescue_min_ratio = 0.75
    engine.father_trigger_rescue_max_nosign_confidence = 0.995
    return engine


def _open_capture(*, simulation_mode: bool, camera_index: int) -> tuple[cv2.VideoCapture, str, bool]:
    if simulation_mode:
        video_path = _find_simulation_video()
        if video_path is None:
            raise RuntimeError("Simulation mode requested but no bundled .mp4 sample was found.")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open simulation video: {video_path}")
        return capture, str(video_path), True

    capture = cv2.VideoCapture(camera_index)
    if capture.isOpened():
        return capture, f"camera:{camera_index}", False

    fallback_video = _find_simulation_video()
    if fallback_video is None:
        raise RuntimeError(f"Unable to open webcam camera index {camera_index}, and no sample video fallback exists.")
    capture.release()
    capture = cv2.VideoCapture(str(fallback_video))
    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open webcam camera index {camera_index}, and unable to open fallback sample video: {fallback_video}"
        )
    return capture, str(fallback_video), True


def _stream_state(prediction) -> tuple[str, float, str]:
    if prediction.emitted_label:
        return prediction.emitted_label, float(prediction.decision.final_confidence), prediction.decision.status
    if prediction.decision.status == "trigger_collecting":
        return "collecting", 0.0, prediction.decision.status
    if prediction.decision.status == "trigger_discard":
        return "discard", 0.0, prediction.decision.status
    return "idle", 0.0, prediction.decision.status


def _annotate_frame(
    *,
    frame,
    source_name: str,
    stream_label: str,
    stream_confidence: float,
    emitted_tokens: list[str],
    fps: float,
):
    annotated = frame.copy()
    line1 = f"Source: {source_name}"
    line2 = f"Stream: {stream_label} ({stream_confidence:.2f})"
    line3 = f"Emitted: {' / '.join(emitted_tokens[-8:]) if emitted_tokens else '-'}"
    line4 = f"FPS: {fps:.1f}  Controls: q=quit"
    cv2.putText(annotated, line1, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(annotated, line2, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(annotated, line3, (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(annotated, line4, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    return annotated


def main() -> None:
    simulation_mode = _env_flag("SIGN_SIMULATION_MODE", default=False)
    headless = _env_flag("SIGN_HEADLESS", default=False)
    mirror_input = _env_flag("SIGN_MIRROR_INPUT", default=True)
    camera_index = _env_int("SIGN_CAMERA_INDEX", 0)
    max_frames = _env_int("SIGN_MAX_FRAMES", 0)

    engine = _build_engine()
    capture = None
    source_name = ""
    used_simulation = simulation_mode
    emitted_tokens: list[str] = []
    last_emitted_frame = -1
    frames_processed = 0
    started_at = time.time()
    fps_started_at = started_at
    fps_counter = 0
    display_fps = 0.0

    try:
        capture, source_name, used_simulation = _open_capture(
            simulation_mode=simulation_mode,
            camera_index=camera_index,
        )
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if mirror_input:
                frame = cv2.flip(frame, 1)

            prediction = engine.process_frame(frame)
            stream_label, stream_confidence, decoder_state = _stream_state(prediction)
            if prediction.emitted_label and prediction.frame_index != last_emitted_frame:
                emitted_tokens.append(prediction.emitted_label)
                last_emitted_frame = prediction.frame_index
                print(
                    json.dumps(
                        {
                            "event": "emit",
                            "frame_index": prediction.frame_index,
                            "label": prediction.emitted_label,
                            "confidence": round(float(prediction.decision.final_confidence), 6),
                            "status": decoder_state,
                        },
                        ensure_ascii=False,
                    )
                )

            frames_processed += 1
            fps_counter += 1
            now = time.time()
            elapsed = max(now - fps_started_at, 1e-6)
            if elapsed >= 0.5:
                display_fps = fps_counter / elapsed
                fps_counter = 0
                fps_started_at = now

            if not headless:
                annotated = _annotate_frame(
                    frame=prediction.display_frame,
                    source_name=source_name,
                    stream_label=stream_label,
                    stream_confidence=stream_confidence,
                    emitted_tokens=emitted_tokens,
                    fps=display_fps,
                )
                cv2.imshow("Realtime Sign Language Recognition", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames > 0 and frames_processed >= max_frames:
                break
    finally:
        if capture is not None:
            capture.release()
        engine.close()
        if not headless:
            cv2.destroyAllWindows()

    duration = max(time.time() - started_at, 1e-6)
    summary = {
        "source": source_name,
        "used_simulation": used_simulation,
        "mirror_input": mirror_input,
        "frames_processed": frames_processed,
        "duration_seconds": round(duration, 3),
        "avg_fps": round(frames_processed / duration, 3),
        "emitted_tokens": emitted_tokens,
        "locked_stack": {
            "pose_local_anchor": "torso_center",
            "mother_fix": True,
            "you_i_fix": True,
            "you_vs_like_fix": True,
            "like_vs_i_fix": True,
            "father_rescue": True,
        },
        "limitations": "want/teacher/father remain training/data-domain residuals (WER ~0.375)",
    }
    DEFAULT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
