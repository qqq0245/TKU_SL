from __future__ import annotations

import numpy as np

from config import CONFIG


LOCATION_VECTOR_NAMES = [
    "left_to_nose",
    "left_to_chin",
    "left_to_mouth",
    "left_to_shoulder_center",
    "left_to_chest_center",
    "left_to_torso_center",
    "right_to_nose",
    "right_to_chin",
    "right_to_mouth",
    "right_to_shoulder_center",
    "right_to_chest_center",
    "right_to_torso_center",
    "left_to_right",
]
FACE_REFERENCE_NAMES = ("nose", "chin", "mouth_center")
HAND_ANCHOR_TO_INDEX = {
    "center": None,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
    "closest_valid_fingertip": "closest_valid_fingertip",
}


def _normalize_face_reference_modes(raw_modes: tuple[str, str, str] | None) -> tuple[str, str, str]:
    if raw_modes is None:
        return FACE_REFERENCE_NAMES
    values = tuple(str(value).strip().lower() for value in raw_modes)
    if len(values) != 3:
        raise ValueError(f"Face reference modes must contain exactly 3 entries: {raw_modes!r}")
    invalid = [value for value in values if value not in FACE_REFERENCE_NAMES]
    if invalid:
        raise ValueError(f"Unsupported face reference mode(s): {invalid!r}")
    return values


def _normalize_face_anchor_modes(
    raw_modes: tuple[str, str, str] | None,
    *,
    default_anchor_mode: str,
) -> tuple[str, str, str]:
    if raw_modes is None:
        return (default_anchor_mode, default_anchor_mode, default_anchor_mode)
    values = tuple(str(value).strip().lower() for value in raw_modes)
    if len(values) != 3:
        raise ValueError(f"Face anchor modes must contain exactly 3 entries: {raw_modes!r}")
    invalid = [value for value in values if value not in HAND_ANCHOR_TO_INDEX]
    if invalid:
        raise ValueError(f"Unsupported face anchor mode(s): {invalid!r}")
    return values


def _relative_vector(source: np.ndarray, target: np.ndarray, source_valid: bool, target_valid: bool = True) -> np.ndarray:
    if not source_valid or not target_valid:
        return np.zeros((3,), dtype=np.float32)
    return (source - target).astype(np.float32)


def _zone_encoding(hand_center: np.ndarray, refs: dict[str, np.ndarray], hand_valid: bool) -> np.ndarray:
    zone = np.zeros((len(CONFIG.location_zone_names),), dtype=np.float32)
    if not hand_valid:
        return zone

    x, y, _ = hand_center
    nose = refs["nose"]
    mouth = refs["mouth_center"]
    shoulder_center = refs["shoulder_center"]
    chest_center = refs["chest_center"]

    if y < nose[1] - 0.15:
        zone[0] = 1.0
    elif y < mouth[1] + 0.08:
        zone[1] = 1.0
    elif y < chest_center[1] + 0.08:
        zone[2] = 1.0
    elif y > chest_center[1] + 0.25:
        zone[6] = 1.0
    elif x < shoulder_center[0] - 0.12:
        zone[3] = 1.0
    elif x > shoulder_center[0] + 0.12:
        zone[4] = 1.0
    else:
        zone[5] = 1.0
    return zone


def _reference_valid(frame_feature: dict[str, np.ndarray], reference_name: str) -> bool:
    if reference_name == "chin":
        return bool(frame_feature["chin_mask"][0] > 0)
    if reference_name == "mouth_center":
        return bool(frame_feature["mouth_mask"][0] > 0)
    return True


def _face_vectors(
    *,
    frame_feature: dict[str, np.ndarray],
    hand_key: str,
    hand_mask_key: str,
    hand_center_key: str,
    hand_valid_key: str,
    refs: dict[str, np.ndarray],
    default_anchor_mode: str,
    face_reference_modes: tuple[str, str, str] | None = None,
    face_anchor_modes: tuple[str, str, str] | None = None,
) -> list[np.ndarray]:
    reference_modes = _normalize_face_reference_modes(face_reference_modes)
    anchor_modes = _normalize_face_anchor_modes(face_anchor_modes, default_anchor_mode=default_anchor_mode)
    hand = frame_feature.get(hand_key, np.zeros((0, 3), dtype=np.float32))
    hand_mask = frame_feature.get(hand_mask_key, np.zeros((0,), dtype=np.float32))
    hand_center = frame_feature[hand_center_key]
    hand_valid = bool(frame_feature[hand_valid_key][0] > 0)
    vectors: list[np.ndarray] = []
    for reference_name, anchor_mode in zip(reference_modes, anchor_modes):
        anchor = resolve_hand_anchor(
            hand=hand,
            hand_mask=hand_mask,
            hand_center=hand_center,
            anchor_mode=anchor_mode,
        )
        vectors.append(
            _relative_vector(
                anchor,
                refs[reference_name],
                hand_valid,
                _reference_valid(frame_feature, reference_name),
            )
        )
    return vectors


def resolve_hand_anchor(
    *,
    hand: np.ndarray,
    hand_mask: np.ndarray,
    hand_center: np.ndarray,
    anchor_mode: str,
) -> np.ndarray:
    anchor_mode = str(anchor_mode).strip().lower()
    if anchor_mode not in HAND_ANCHOR_TO_INDEX:
        raise ValueError(f"Unsupported hand anchor mode: {anchor_mode}")
    anchor_index = HAND_ANCHOR_TO_INDEX[anchor_mode]
    if anchor_index is None:
        return hand_center.astype(np.float32)
    if anchor_index == "closest_valid_fingertip":
        fingertip_indices = (4, 8, 12, 16, 20)
        for fingertip_index in fingertip_indices:
            if hand.shape[0] > fingertip_index and hand_mask.shape[0] > fingertip_index and hand_mask[fingertip_index] > 0:
                return hand[fingertip_index].astype(np.float32)
        return hand_center.astype(np.float32)
    if hand.shape[0] <= anchor_index or hand_mask.shape[0] <= anchor_index or hand_mask[anchor_index] <= 0:
        return hand_center.astype(np.float32)
    return hand[anchor_index].astype(np.float32)


def build_location_features(
    frame_feature: dict[str, np.ndarray],
    *,
    left_anchor_mode: str = "center",
    right_anchor_mode: str = "center",
    left_face_reference_modes: tuple[str, str, str] | None = None,
    left_face_anchor_modes: tuple[str, str, str] | None = None,
) -> dict[str, np.ndarray]:
    refs = frame_feature["reference_points"]
    left_center = resolve_hand_anchor(
        hand=frame_feature.get("left_hand", np.zeros((0, 3), dtype=np.float32)),
        hand_mask=frame_feature.get("left_hand_mask", np.zeros((0,), dtype=np.float32)),
        hand_center=frame_feature["left_hand_center"],
        anchor_mode=left_anchor_mode,
    )
    right_center = resolve_hand_anchor(
        hand=frame_feature.get("right_hand", np.zeros((0, 3), dtype=np.float32)),
        hand_mask=frame_feature.get("right_hand_mask", np.zeros((0,), dtype=np.float32)),
        hand_center=frame_feature["right_hand_center"],
        anchor_mode=right_anchor_mode,
    )
    left_valid = bool(frame_feature["left_hand_valid"][0] > 0)
    right_valid = bool(frame_feature["right_hand_valid"][0] > 0)
    chin_valid = bool(frame_feature["chin_mask"][0] > 0)
    mouth_valid = bool(frame_feature["mouth_mask"][0] > 0)

    vectors = [
        *_face_vectors(
            frame_feature=frame_feature,
            hand_key="left_hand",
            hand_mask_key="left_hand_mask",
            hand_center_key="left_hand_center",
            hand_valid_key="left_hand_valid",
            refs=refs,
            default_anchor_mode=left_anchor_mode,
            face_reference_modes=left_face_reference_modes,
            face_anchor_modes=left_face_anchor_modes,
        ),
        _relative_vector(left_center, refs["shoulder_center"], left_valid),
        _relative_vector(left_center, refs["chest_center"], left_valid),
        _relative_vector(left_center, refs["torso_center"], left_valid),
        _relative_vector(right_center, refs["nose"], right_valid),
        _relative_vector(right_center, refs["chin"], right_valid, chin_valid),
        _relative_vector(right_center, refs["mouth_center"], right_valid, mouth_valid),
        _relative_vector(right_center, refs["shoulder_center"], right_valid),
        _relative_vector(right_center, refs["chest_center"], right_valid),
        _relative_vector(right_center, refs["torso_center"], right_valid),
        _relative_vector(left_center, right_center, left_valid and right_valid),
    ]

    left_zone = _zone_encoding(left_center, refs, left_valid) if CONFIG.use_zone_encoding else np.zeros((0,), dtype=np.float32)
    right_zone = _zone_encoding(right_center, refs, right_valid) if CONFIG.use_zone_encoding else np.zeros((0,), dtype=np.float32)
    validity = np.array(
        [
            float(left_valid),
            float(right_valid),
            float(chin_valid),
            float(mouth_valid),
        ],
        dtype=np.float32,
    )

    feature_vector = np.concatenate(
        [*(vector.reshape(-1) for vector in vectors), left_zone, right_zone, validity],
        axis=0,
    ).astype(np.float32)

    return {
        "feature_vector": feature_vector,
        "left_zone": left_zone,
        "right_zone": right_zone,
        "validity": validity,
    }
