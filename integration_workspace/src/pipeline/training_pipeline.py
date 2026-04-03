from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import CONFIG
from src.dataset.dataset_reader import (
    SignSequenceDataset,
    create_label_mapping,
    filter_records_by_feature_mode,
    load_index,
    load_split_records,
    split_records,
    validate_feature_mode,
)
from src.landmarks.feature_builder import get_feature_spec
from src.models.lstm_model import LSTMBaseline
from src.utils.logger import get_logger
from src.utils.paths import ensure_dir


LOGGER = get_logger(__name__)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    total_accuracy = 0.0
    total_count = 0

    for sequences, labels in tqdm(loader, leave=False):
        sequences = sequences.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_accuracy += _accuracy(logits, labels) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1), total_accuracy / max(total_count, 1)


def _build_class_weights(records: list[dict], label_to_index: dict[str, int]) -> torch.Tensor:
    counts = Counter(record["class_label"] for record in records)
    weights = torch.ones(len(label_to_index), dtype=torch.float32)
    for label, index in label_to_index.items():
        weights[index] = 1.0 / max(counts.get(label, 1), 1)
    weights = weights / weights.mean()
    return weights


def _build_train_loader(dataset: SignSequenceDataset, train_records: list[dict], label_to_index: dict[str, int]) -> DataLoader:
    if not CONFIG.use_weighted_sampler:
        return DataLoader(dataset, batch_size=CONFIG.batch_size, shuffle=True)
    class_weights = _build_class_weights(train_records, label_to_index)
    sample_weights = [float(class_weights[label_to_index[record["class_label"]]]) for record in train_records]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=CONFIG.batch_size, sampler=sampler)


def train_lstm_baseline(
    data_dir: str | None = None,
    train_split_csv: str | None = None,
    val_split_csv: str | None = None,
) -> None:
    resolved_data_dir = Path(data_dir) if data_dir is not None else CONFIG.processed_dir

    if train_split_csv:
        train_records = load_split_records(Path(train_split_csv))
        val_records = load_split_records(Path(val_split_csv)) if val_split_csv else []
        combined_records = train_records + val_records
        if len(combined_records) < 2:
            raise RuntimeError("Need at least 2 samples before training.")
        validate_feature_mode(combined_records, CONFIG.feature_mode)
        label_to_index = create_label_mapping(combined_records)
    else:
        records = load_index(resolved_data_dir / "index.jsonl")
        if len(records) < 2:
            raise RuntimeError("Need at least 2 samples before training.")
        compatible_records = filter_records_by_feature_mode(records, CONFIG.feature_mode)
        if compatible_records and len(compatible_records) != len(records):
            LOGGER.warning(
                "Found mixed feature modes in dataset. Using %s compatible samples out of %s total samples.",
                len(compatible_records),
                len(records),
            )
            records = compatible_records
        validate_feature_mode(records, CONFIG.feature_mode)
        label_to_index = create_label_mapping(records)
        train_records, val_records = split_records(records, CONFIG.val_ratio, CONFIG.random_seed)

    index_to_label = {index: label for label, index in label_to_index.items()}

    train_dataset = SignSequenceDataset(
        resolved_data_dir,
        train_records,
        label_to_index,
        training=True,
        use_spatial_translation_aug=CONFIG.use_spatial_translation_aug,
        random_seed=CONFIG.random_seed,
    )
    val_dataset = SignSequenceDataset(resolved_data_dir, val_records, label_to_index, training=False)

    sample_sequence, _ = train_dataset[0]
    input_dim = int(sample_sequence.shape[-1])

    train_loader = _build_train_loader(train_dataset, train_records, label_to_index)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, shuffle=False) if val_records else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMBaseline(
        input_dim=input_dim,
        hidden_size=CONFIG.hidden_size,
        num_layers=CONFIG.num_layers,
        num_classes=len(label_to_index),
        dropout=CONFIG.dropout,
    ).to(device)

    class_weights = _build_class_weights(train_records, label_to_index) if CONFIG.use_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )

    ensure_dir(CONFIG.models_dir)
    checkpoint_path = CONFIG.models_dir / "lstm_baseline.pt"
    best_val_accuracy = -1.0

    LOGGER.info(
        "Training samples=%s validation samples=%s input_dim=%s feature_mode=%s data_dir=%s weighted_loss=%s weighted_sampler=%s spatial_aug=%s weight_decay=%.6g",
        len(train_dataset),
        len(val_dataset),
        input_dim,
        CONFIG.feature_mode,
        resolved_data_dir,
        CONFIG.use_weighted_loss,
        CONFIG.use_weighted_sampler,
        CONFIG.use_spatial_translation_aug,
        CONFIG.weight_decay,
    )

    for epoch in range(1, CONFIG.num_epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        if val_loader is not None:
            val_loss, val_accuracy = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        else:
            val_loss, val_accuracy = 0.0, 0.0

        LOGGER.info(
            "Epoch %s/%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            CONFIG.num_epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        if val_loader is None or val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": len(label_to_index),
                    "label_to_index": label_to_index,
                    "index_to_label": index_to_label,
                    "hidden_size": CONFIG.hidden_size,
                    "num_layers": CONFIG.num_layers,
                    "dropout": CONFIG.dropout,
                    "feature_mode": CONFIG.feature_mode,
                    "feature_spec": get_feature_spec(CONFIG.feature_mode),
                    "use_spatial_translation_aug": CONFIG.use_spatial_translation_aug,
                    "spatial_translation_min_offset": CONFIG.spatial_translation_min_offset,
                    "spatial_translation_max_offset": CONFIG.spatial_translation_max_offset,
                    "use_weighted_loss": CONFIG.use_weighted_loss,
                    "weight_decay": CONFIG.weight_decay,
                },
                checkpoint_path,
            )

    with (CONFIG.models_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "input_dim": input_dim,
                "labels": label_to_index,
                "data_dir": str(resolved_data_dir),
                "checkpoint": str(checkpoint_path),
                "feature_mode": CONFIG.feature_mode,
                "feature_spec": get_feature_spec(CONFIG.feature_mode),
                "use_spatial_translation_aug": CONFIG.use_spatial_translation_aug,
                "spatial_translation_min_offset": CONFIG.spatial_translation_min_offset,
                "spatial_translation_max_offset": CONFIG.spatial_translation_max_offset,
                "use_weighted_loss": CONFIG.use_weighted_loss,
                "weight_decay": CONFIG.weight_decay,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    LOGGER.info("Saved checkpoint to %s", checkpoint_path)
