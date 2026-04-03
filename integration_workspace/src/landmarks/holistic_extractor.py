from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config import CONFIG


@dataclass
class FrameLandmarks:
    left_hand: np.ndarray
    right_hand: np.ndarray
    pose: np.ndarray
    mouth_center: np.ndarray
    chin: np.ndarray
    left_hand_mask: np.ndarray
    right_hand_mask: np.ndarray
    pose_mask: np.ndarray
    mouth_mask: np.ndarray
    chin_mask: np.ndarray


class HolisticExtractor:
    def __init__(
        self,
        min_detection_confidence: float = CONFIG.min_detection_confidence,
        min_tracking_confidence: float = CONFIG.min_tracking_confidence,
    ) -> None:
        self.pose_indices = list(CONFIG.pose_indices)
        self.face_reference_indices = CONFIG.face_reference_indices
        self.mp_holistic = None
        self.mp_drawing = None
        self.model = None
        self.task_model = None
        self.using_tasks_api = False
        self.closed = False

        if hasattr(mp, "solutions"):
            self.mp_holistic = mp.solutions.holistic
            self.mp_drawing = mp.solutions.drawing_utils
            self.model = self.mp_holistic.Holistic(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            return

        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.holistic_landmarker import (
            HolisticLandmarker,
            HolisticLandmarkerOptions,
        )
        from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

        model_override = os.getenv("SIGN_HOLISTIC_TASK_MODEL")
        model_path = Path(model_override) if model_override else CONFIG.project_root / "artifacts" / "mediapipe" / "holistic_landmarker.task"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Holistic task model not found: {model_path}. "
                "Set SIGN_HOLISTIC_TASK_MODEL to a valid holistic_landmarker.task path."
            )

        self.task_image_cls = Image
        self.task_image_format = ImageFormat
        self.task_model = HolisticLandmarker.create_from_options(
            HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_landmarks_confidence=min_tracking_confidence,
                min_hand_landmarks_confidence=min_tracking_confidence,
                output_face_blendshapes=False,
                output_segmentation_mask=False,
            )
        )
        self.using_tasks_api = True

    def extract(self, frame_bgr: np.ndarray) -> tuple[FrameLandmarks, object]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.using_tasks_api:
            image = self.task_image_cls(image_format=self.task_image_format.SRGB, data=np.ascontiguousarray(rgb))
            results = self.task_model.detect(image)
        else:
            rgb.flags.writeable = False
            results = self.model.process(rgb)

        left_hand, left_mask = self._extract_hand(results.left_hand_landmarks)
        right_hand, right_mask = self._extract_hand(results.right_hand_landmarks)
        pose, pose_mask = self._extract_pose(results.pose_landmarks)
        mouth_center, chin, mouth_mask, chin_mask = self._extract_face_references(results.face_landmarks)

        return (
            FrameLandmarks(
                left_hand=left_hand,
                right_hand=right_hand,
                pose=pose,
                mouth_center=mouth_center,
                chin=chin,
                left_hand_mask=left_mask,
                right_hand_mask=right_mask,
                pose_mask=pose_mask,
                mouth_mask=mouth_mask,
                chin_mask=chin_mask,
            ),
            results,
        )

    def draw(self, frame_bgr: np.ndarray, results: object) -> np.ndarray:
        output = frame_bgr.copy()
        if self.using_tasks_api:
            return output
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(output, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(output, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(output, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        return output

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.task_model is not None:
                self.task_model.close()
            elif self.model is not None:
                self.model.close()
        except Exception:
            # MediaPipe Tasks can already be in a failed/closed state after graph errors.
            pass

    def _extract_hand(self, hand_landmarks) -> tuple[np.ndarray, np.ndarray]:
        points = np.zeros((CONFIG.left_hand_nodes, CONFIG.channels), dtype=np.float32)
        mask = np.zeros((CONFIG.left_hand_nodes,), dtype=np.float32)
        landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
        if landmarks:
            for idx, landmark in enumerate(landmarks[: CONFIG.left_hand_nodes]):
                points[idx] = [landmark.x, landmark.y, landmark.z]
                mask[idx] = 1.0
        return points, mask

    def _extract_pose(self, pose_landmarks) -> tuple[np.ndarray, np.ndarray]:
        points = np.zeros((len(self.pose_indices), CONFIG.channels), dtype=np.float32)
        mask = np.zeros((len(self.pose_indices),), dtype=np.float32)
        landmarks = pose_landmarks.landmark if hasattr(pose_landmarks, "landmark") else pose_landmarks
        if landmarks:
            for output_idx, pose_idx in enumerate(self.pose_indices):
                landmark = landmarks[pose_idx]
                points[output_idx] = [landmark.x, landmark.y, landmark.z]
                mask[output_idx] = 1.0
        return points, mask

    def _extract_face_references(self, face_landmarks) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mouth_center = np.zeros((3,), dtype=np.float32)
        chin = np.zeros((3,), dtype=np.float32)
        mouth_mask = np.zeros((1,), dtype=np.float32)
        chin_mask = np.zeros((1,), dtype=np.float32)

        landmarks = face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks
        if not landmarks:
            return mouth_center, chin, mouth_mask, chin_mask

        mouth_points = []
        for name in ("mouth_upper", "mouth_lower", "mouth_left", "mouth_right"):
            idx = self.face_reference_indices[name]
            landmark = landmarks[idx]
            mouth_points.append([landmark.x, landmark.y, landmark.z])
        mouth_center = np.mean(np.asarray(mouth_points, dtype=np.float32), axis=0)
        mouth_mask[0] = 1.0

        chin_idx = self.face_reference_indices["chin"]
        chin_landmark = landmarks[chin_idx]
        chin = np.asarray([chin_landmark.x, chin_landmark.y, chin_landmark.z], dtype=np.float32)
        chin_mask[0] = 1.0
        return mouth_center, chin, mouth_mask, chin_mask
