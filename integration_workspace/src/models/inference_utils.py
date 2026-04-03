from __future__ import annotations

from pathlib import Path

import torch

from config import CONFIG
from src.models.lstm_model import LSTMBaseline


def load_lstm_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LSTMBaseline(
        input_dim=checkpoint["input_dim"],
        hidden_size=checkpoint.get("hidden_size", CONFIG.hidden_size),
        num_layers=checkpoint.get("num_layers", CONFIG.num_layers),
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint.get("dropout", CONFIG.dropout),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def apply_confidence_threshold(pred_label: str, confidence: float, threshold: float | None = None) -> tuple[str, str]:
    threshold = CONFIG.confidence_threshold if threshold is None else threshold
    if not CONFIG.enable_confidence_threshold:
        return pred_label, "accepted"
    if pred_label == CONFIG.no_sign_label:
        return pred_label, "accepted"
    if confidence < threshold:
        return CONFIG.unknown_label, "low_confidence"
    return pred_label, "accepted"


@torch.no_grad()
def predict_sequence(model, sequence_tensor: torch.Tensor, index_to_label: dict[int, str]):
    logits = model(sequence_tensor)
    probs = torch.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probs, dim=-1)
    return index_to_label[int(prediction.item())], float(confidence.item()), probs.squeeze(0).cpu().numpy()
