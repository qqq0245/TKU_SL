from __future__ import annotations

import json
import time

import cv2
import torch

from config import CONFIG
from src.camera.webcam_stream import WebcamStream
from src.dataset.dataset_writer import DatasetWriter
from src.dataset.sequence_builder import SequenceBuilder
from src.landmarks.feature_builder import build_frame_feature, get_feature_spec
from src.landmarks.holistic_extractor import HolisticExtractor
from src.models.inference_utils import apply_confidence_threshold, load_lstm_checkpoint, predict_sequence
from src.models.inference_utils_multibranch import load_multibranch_checkpoint, predict_multibranch_sequence
from src.pipeline.gcn_export_adapter import export_sample_to_gcn_dict
from src.pipeline.inference_postprocess import InferencePostprocessor
from src.utils.labels import ensure_label_registered
from src.utils.logger import get_logger
from src.utils.paths import ensure_dir


LOGGER = get_logger(__name__)


def _sample_id(label: str, sequence_index: int, sample_prefix: str = "") -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{sample_prefix}_" if sample_prefix else ""
    return f"{prefix}{label}_{timestamp}_{sequence_index:03d}"


def run_capture_session(
    label: str,
    num_sequences: int,
    sequence_length: int,
    camera_index: int,
    sample_prefix: str = "",
) -> None:
    ensure_dir(CONFIG.processed_dir)
    ensure_dir(CONFIG.exports_dir)
    ensure_label_registered(CONFIG.processed_dir, label)

    writer = DatasetWriter(CONFIG.processed_dir)
    extractor = HolisticExtractor()
    sequence_index = 0

    with WebcamStream(camera_index) as webcam:
        while sequence_index < num_sequences:
            builder = SequenceBuilder(sequence_length)
            frame_valid_builder = SequenceBuilder(sequence_length)
            prev_feature = None
            recording = False

            while True:
                frame = webcam.read()
                frame_landmarks, results = extractor.extract(frame)
                built = build_frame_feature(
                    left_hand=frame_landmarks.left_hand,
                    right_hand=frame_landmarks.right_hand,
                    pose=frame_landmarks.pose,
                    mouth_center=frame_landmarks.mouth_center,
                    chin=frame_landmarks.chin,
                    left_hand_mask=frame_landmarks.left_hand_mask,
                    right_hand_mask=frame_landmarks.right_hand_mask,
                    pose_mask=frame_landmarks.pose_mask,
                    mouth_mask=frame_landmarks.mouth_mask,
                    chin_mask=frame_landmarks.chin_mask,
                    prev_feature=prev_feature,
                    feature_mode=CONFIG.feature_mode,
                )
                prev_feature = built
                display = extractor.draw(frame, results)

                if recording:
                    builder.append(built["feature_vector"])
                    frame_valid_builder.append(built["frame_valid_mask"])

                cv2.putText(display, f"Label={label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
                cv2.putText(
                    display,
                    f"Mode={CONFIG.feature_mode} Sample {sequence_index + 1}/{num_sequences} Frames {len(builder)}/{sequence_length}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(display, "Press r to record, q to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("Capture Sign Sequences", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    extractor.close()
                    cv2.destroyAllWindows()
                    return
                if not recording and key == ord("r"):
                    recording = True
                    LOGGER.info("Recording sample %s for label=%s", sequence_index + 1, label)

                if recording and builder.is_full():
                    sample_id = _sample_id(label, sequence_index, sample_prefix)
                    sequence = builder.sliding_window()
                    metadata = {
                        "label": label,
                        "sample_id": sample_id,
                        "sequence_length": sequence_length,
                        "feature_dim": int(sequence.shape[1]),
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "capture_source": "webcam",
                        "normalization_origin": "torso_center",
                        "pose_indices": list(CONFIG.pose_indices),
                        "feature_mode": CONFIG.feature_mode,
                        "feature_spec": built["feature_spec"],
                    }
                    writer.write_sample(
                        sample_id=sample_id,
                        class_label=label,
                        sequence=sequence,
                        metadata=metadata,
                        frame_valid_mask=frame_valid_builder.sliding_window(),
                    )

                    preview = export_sample_to_gcn_dict(sequence=sequence, metadata=metadata)
                    with (CONFIG.exports_dir / f"{sample_id}_gcn_preview.json").open("w", encoding="utf-8") as handle:
                        json.dump(preview, handle, ensure_ascii=False, indent=2)

                    sequence_index += 1
                    break

    extractor.close()
    cv2.destroyAllWindows()


def run_realtime_inference(
    checkpoint_path: str,
    sequence_length: int,
    camera_index: int,
    confidence_threshold: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_lstm_checkpoint(checkpoint_path, device)
    extractor = HolisticExtractor()
    builder = SequenceBuilder(sequence_length)
    frame_valid_builder = SequenceBuilder(sequence_length)
    prev_feature = None
    postprocessor = InferencePostprocessor()
    if checkpoint is not None:
        checkpoint_mode = checkpoint.get("feature_mode")
        checkpoint_dim = checkpoint.get("input_dim")
        current_dim = get_feature_spec(CONFIG.feature_mode)["total_dim"]
        if checkpoint_mode != CONFIG.feature_mode:
            raise RuntimeError(
                f"Checkpoint feature_mode={checkpoint_mode} does not match current config feature_mode={CONFIG.feature_mode}"
            )
        if checkpoint_dim != current_dim:
            raise RuntimeError(f"Checkpoint input_dim={checkpoint_dim} does not match current feature_dim={current_dim}")
        index_to_label = checkpoint["index_to_label"]
    else:
        index_to_label = None

    with WebcamStream(camera_index) as webcam:
        while True:
            frame = webcam.read()
            frame_landmarks, results = extractor.extract(frame)
            built = build_frame_feature(
                left_hand=frame_landmarks.left_hand,
                right_hand=frame_landmarks.right_hand,
                pose=frame_landmarks.pose,
                mouth_center=frame_landmarks.mouth_center,
                chin=frame_landmarks.chin,
                left_hand_mask=frame_landmarks.left_hand_mask,
                right_hand_mask=frame_landmarks.right_hand_mask,
                pose_mask=frame_landmarks.pose_mask,
                mouth_mask=frame_landmarks.mouth_mask,
                chin_mask=frame_landmarks.chin_mask,
                prev_feature=prev_feature,
                feature_mode=CONFIG.feature_mode,
            )
            prev_feature = built
            builder.append(built["feature_vector"])
            frame_valid_builder.append(built["frame_valid_mask"])
            display = extractor.draw(frame, results)

            if model is None or index_to_label is None:
                final_line = "Prediction: no_checkpoint"
                raw_line = "Raw: n/a"
                status_line = "Status: missing_checkpoint"
                color = (0, 255, 255)
            else:
                final_line = f"Prediction: waiting ({len(builder)}/{sequence_length})"
                raw_line = "Raw: n/a"
                status_line = "Status: collecting"
                color = (255, 255, 255)
                sequence = builder.sliding_window()
                if sequence is not None:
                    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
                    raw_label, raw_confidence, _ = predict_sequence(model, tensor, index_to_label)
                    thresholded_label, threshold_status = apply_confidence_threshold(raw_label, raw_confidence, confidence_threshold)
                    decision = postprocessor.decide(
                        raw_label=raw_label,
                        raw_confidence=raw_confidence,
                        frame_valid_mask=built["frame_valid_mask"],
                        frame_valid_sequence=frame_valid_builder.sliding_window(),
                        thresholded_label=thresholded_label,
                        threshold_status=threshold_status,
                    )
                    final_line = f"Prediction: {decision.final_label} ({decision.final_confidence:.2f})"
                    raw_line = f"Raw: {decision.raw_label} ({decision.raw_confidence:.2f})"
                    status_line = f"Status: {decision.status} valid={decision.valid_ratio:.2f}"
                    if decision.final_label == CONFIG.unknown_label:
                        color = (0, 255, 255)
                    elif decision.final_label == CONFIG.no_sign_label:
                        color = (255, 200, 0)
                    else:
                        color = (50, 255, 50)

            cv2.putText(display, final_line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, raw_line, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, status_line, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Mode={CONFIG.feature_mode}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, "Press q to quit", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Realtime Inference", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    extractor.close()
    cv2.destroyAllWindows()


def run_realtime_inference_multibranch(
    checkpoint_path: str,
    sequence_length: int,
    camera_index: int,
    confidence_threshold: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_multibranch_checkpoint(checkpoint_path, device)
    extractor = HolisticExtractor()
    builder = SequenceBuilder(sequence_length)
    frame_valid_builder = SequenceBuilder(sequence_length)
    prev_feature = None
    postprocessor = InferencePostprocessor()

    if checkpoint is not None:
        checkpoint_mode = checkpoint.get("feature_mode")
        checkpoint_dim = checkpoint.get("input_dim")
        checkpoint_branch = checkpoint.get("skeleton_branch_type")
        if checkpoint_branch is None:
            checkpoint_branch = "gcn" if checkpoint.get("use_gcn_skeleton", False) else "lstm"
        current_dim = get_feature_spec(CONFIG.feature_mode)["total_dim"]
        if checkpoint_mode != CONFIG.feature_mode:
            raise RuntimeError(
                f"Checkpoint feature_mode={checkpoint_mode} does not match current config feature_mode={CONFIG.feature_mode}"
            )
        if checkpoint_dim != current_dim:
            raise RuntimeError(f"Checkpoint input_dim={checkpoint_dim} does not match current feature_dim={current_dim}")
        index_to_label = checkpoint["index_to_label"]
    else:
        index_to_label = None
        checkpoint_branch = "unknown"

    with WebcamStream(camera_index) as webcam:
        while True:
            frame = webcam.read()
            frame_landmarks, results = extractor.extract(frame)
            built = build_frame_feature(
                left_hand=frame_landmarks.left_hand,
                right_hand=frame_landmarks.right_hand,
                pose=frame_landmarks.pose,
                mouth_center=frame_landmarks.mouth_center,
                chin=frame_landmarks.chin,
                left_hand_mask=frame_landmarks.left_hand_mask,
                right_hand_mask=frame_landmarks.right_hand_mask,
                pose_mask=frame_landmarks.pose_mask,
                mouth_mask=frame_landmarks.mouth_mask,
                chin_mask=frame_landmarks.chin_mask,
                prev_feature=prev_feature,
                feature_mode=CONFIG.feature_mode,
            )
            prev_feature = built
            builder.append(built["feature_vector"])
            frame_valid_builder.append(built["frame_valid_mask"])
            display = extractor.draw(frame, results)

            if model is None or index_to_label is None:
                final_line = "Prediction: no_checkpoint"
                raw_line = "Raw: n/a"
                status_line = "Status: missing_checkpoint"
                color = (0, 255, 255)
            else:
                final_line = f"Prediction: waiting ({len(builder)}/{sequence_length})"
                raw_line = "Raw: n/a"
                status_line = "Status: collecting"
                color = (255, 255, 255)
                sequence = builder.sliding_window()
                if sequence is not None:
                    tensor = torch.from_numpy(sequence).unsqueeze(0).to(device)
                    raw_label, raw_confidence, _ = predict_multibranch_sequence(
                        model,
                        tensor,
                        index_to_label,
                        CONFIG.feature_mode,
                    )
                    thresholded_label, threshold_status = apply_confidence_threshold(raw_label, raw_confidence, confidence_threshold)
                    decision = postprocessor.decide(
                        raw_label=raw_label,
                        raw_confidence=raw_confidence,
                        frame_valid_mask=built["frame_valid_mask"],
                        frame_valid_sequence=frame_valid_builder.sliding_window(),
                        thresholded_label=thresholded_label,
                        threshold_status=threshold_status,
                    )
                    final_line = f"Prediction: {decision.final_label} ({decision.final_confidence:.2f})"
                    raw_line = f"Raw: {decision.raw_label} ({decision.raw_confidence:.2f})"
                    status_line = f"Status: {decision.status} valid={decision.valid_ratio:.2f}"
                    if decision.final_label == CONFIG.unknown_label:
                        color = (0, 255, 255)
                    elif decision.final_label == CONFIG.no_sign_label:
                        color = (255, 200, 0)
                    else:
                        color = (50, 255, 50)

            cv2.putText(display, final_line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, raw_line, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, status_line, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(
                display,
                f"Mode={CONFIG.feature_mode} SkeletonBranch={checkpoint_branch}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(display, "Press q to quit", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Realtime Multibranch Inference", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    extractor.close()
    cv2.destroyAllWindows()
