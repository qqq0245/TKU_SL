from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DisambiguationContext:
    probabilities: dict[str, float]
    top_candidates: list[tuple[str, float]]
    built_features: dict[str, np.ndarray]


@dataclass(frozen=True)
class DisambiguationResult:
    probabilities: dict[str, float]
    final_label: str
    final_confidence: float
    applied: bool
    rule_name: str
    notes: str


class FatherMotherLocationRule:
    target_labels = ("father", "mother")

    def apply(self, context: DisambiguationContext) -> DisambiguationResult | None:
        top_two_labels = {label for label, _score in context.top_candidates[:2]}
        if not top_two_labels.intersection(self.target_labels):
            return None

        built = context.built_features
        right_valid = bool(built["right_hand_valid"][0] > 0)
        left_valid = bool(built["left_hand_valid"][0] > 0)
        nose_valid = bool(built["pose_mask"][0] > 0)
        if not nose_valid or (not right_valid and not left_valid):
            return None

        hand = built["right_hand"] if right_valid else built["left_hand"]
        hand_mask = built["right_hand_mask"] if right_valid else built["left_hand_mask"]
        if hand_mask[0] <= 0:
            return None

        refs = built["reference_points"]
        wrist_y = float(hand[0][1])
        nose_y = float(refs["nose"][1])
        chin_valid = bool(built["chin_mask"][0] > 0)
        chin_y = float(refs["chin"][1]) if chin_valid else nose_y + 0.18
        face_span = max(chin_y - nose_y, 0.12)
        relative_y = wrist_y - nose_y
        father_threshold = face_span * 0.60
        mother_threshold = face_span * 1.00

        if relative_y <= father_threshold:
            forced_label = "father"
        elif relative_y >= mother_threshold:
            forced_label = "mother"
        else:
            return None

        updated = dict(context.probabilities)
        forced_score = max(updated.get("father", 0.0), updated.get("mother", 0.0))
        fallback_label = "mother" if forced_label == "father" else "father"
        updated[forced_label] = forced_score
        updated[fallback_label] = min(updated.get("father", 0.0), updated.get("mother", 0.0))
        total = float(sum(updated.values()))
        if total > 1e-8:
            updated = {label: float(score / total) for label, score in updated.items()}
        return DisambiguationResult(
            probabilities=updated,
            final_label=forced_label,
            final_confidence=float(updated.get(forced_label, 0.0)),
            applied=True,
            rule_name="father_mother_location",
            notes=(
                f"relative_y={relative_y:.3f} "
                f"father<= {father_threshold:.3f} mother>= {mother_threshold:.3f}"
            ),
        )


class ClassSpecificDisambiguator:
    def __init__(self) -> None:
        self.rules = [FatherMotherLocationRule()]

    def apply(
        self,
        probabilities: dict[str, float],
        built_features: dict[str, np.ndarray],
    ) -> DisambiguationResult:
        ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        context = DisambiguationContext(
            probabilities=probabilities,
            top_candidates=ranked[:3],
            built_features=built_features,
        )
        for rule in self.rules:
            result = rule.apply(context)
            if result is not None and result.applied:
                return result
        top_label, top_score = ranked[0] if ranked else ("collecting", 0.0)
        return DisambiguationResult(
            probabilities=dict(probabilities),
            final_label=top_label,
            final_confidence=float(top_score),
            applied=False,
            rule_name="",
            notes="",
        )
