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
from src.models.feature_slices import get_feature_slices, require_multibranch_mode, split_feature_tensor
from src.models.multibranch_model import MultiBranchSignModel
from src.utils.logger import get_logger
from src.utils.paths import ensure_dir


LOGGER = get_logger(__name__)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


def _run_epoch(model, loader, criterion, optimizer, device, feature_mode: str, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    total_accuracy = 0.0
    total_count = 0

    for sequences, labels in tqdm(loader, leave=False):
        sequences = sequences.to(device)
        labels = labels.to(device)
        streams = split_feature_tensor(sequences, feature_mode)

        if train:
            optimizer.zero_grad()

        logits = model(
            streams["skeleton_stream"],
            streams["location_stream"],
            streams["motion_stream"],
        )
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


def _build_scheduler(optimizer: torch.optim.Optimizer):
    if CONFIG.lr_scheduler_type == "none":
        return None
    if CONFIG.lr_scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(CONFIG.num_epochs, 1),
            eta_min=CONFIG.lr_scheduler_min_lr,
        )
    if CONFIG.lr_scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=CONFIG.lr_scheduler_factor,
            patience=CONFIG.lr_scheduler_patience,
            min_lr=CONFIG.lr_scheduler_min_lr,
        )
    raise ValueError(f"Unsupported lr_scheduler_type={CONFIG.lr_scheduler_type}")


def train_multibranch_baseline(
    data_dir: str | None = None,
    train_split_csv: str | None = None,
    val_split_csv: str | None = None,
) -> None:
    require_multibranch_mode(CONFIG.feature_mode)
    resolved_data_dir = Path(data_dir) if data_dir is not None else CONFIG.processed_dir

    if train_split_csv:
        train_records = load_split_records(Path(train_split_csv))
        val_records = load_split_records(Path(val_split_csv)) if val_split_csv else []
        combined_records = train_records + val_records
        validate_feature_mode(combined_records, CONFIG.feature_mode)
    else:
        records = load_index(resolved_data_dir / "index.jsonl")
        compatible_records = filter_records_by_feature_mode(records, CONFIG.feature_mode)
        if len(compatible_records) != len(records):
            LOGGER.warning(
                "Found mixed feature modes in dataset. Using %s compatible samples out of %s total samples.",
                len(compatible_records),
                len(records),
            )
        records = compatible_records
        if len(records) < 2:
            raise RuntimeError("Need at least 2 compatible samples before multibranch training.")
        validate_feature_mode(records, CONFIG.feature_mode)
        train_records, val_records = split_records(records, CONFIG.val_ratio, CONFIG.random_seed)

    label_to_index = create_label_mapping(train_records + val_records)
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

    slices = get_feature_slices(CONFIG.feature_mode)
    feature_spec = get_feature_spec(CONFIG.feature_mode)

    train_loader = _build_train_loader(train_dataset, train_records, label_to_index)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, shuffle=False) if val_records else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiBranchSignModel(
        skeleton_dim=slices.skeleton_dim,
        location_dim=slices.location_dim,
        motion_dim=slices.motion_dim,
        num_classes=len(label_to_index),
        skeleton_hidden_dim=CONFIG.skeleton_branch_hidden_dim,
        location_hidden_dim=CONFIG.location_branch_hidden_dim,
        motion_hidden_dim=CONFIG.motion_branch_hidden_dim,
        fusion_hidden_dim=CONFIG.fusion_hidden_dim,
        dropout=CONFIG.multibranch_dropout,
        bidirectional=CONFIG.multibranch_bidirectional,
        skeleton_branch_type=CONFIG.skeleton_branch_type,
        gcn_hidden_dims=CONFIG.gcn_hidden_dims,
        stgcn_hidden_dim=CONFIG.stgcn_hidden_dim,
        stgcn_block_channels=CONFIG.stgcn_block_channels,
        stgcn_dropout=CONFIG.stgcn_dropout,
        stgcn_use_residual=CONFIG.stgcn_use_residual,
        stgcn_temporal_kernel_size=CONFIG.stgcn_temporal_kernel_size,
        location_dropout_prob=CONFIG.location_dropout_prob,
    ).to(device)

    class_weights = _build_class_weights(train_records, label_to_index) if CONFIG.use_weighted_loss else None
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )
    scheduler = _build_scheduler(optimizer)

    ensure_dir(CONFIG.models_dir)
    best_checkpoint_path = CONFIG.multibranch_checkpoint_path
    last_checkpoint_path = CONFIG.models_dir / "multibranch_baseline_last.pt"
    best_val_accuracy = -1.0

    LOGGER.info(
        "Multibranch training feature_mode=%s feature_dim=%s skeleton_dim=%s location_dim=%s motion_dim=%s skeleton_branch=%s data_dir=%s weighted_loss=%s weighted_sampler=%s spatial_aug=%s location_dropout_prob=%.2f weight_decay=%.6g scheduler=%s",
        CONFIG.feature_mode,
        input_dim,
        slices.skeleton_dim,
        slices.location_dim,
        slices.motion_dim,
        CONFIG.skeleton_branch_type,
        resolved_data_dir,
        CONFIG.use_weighted_loss,
        CONFIG.use_weighted_sampler,
        CONFIG.use_spatial_translation_aug,
        CONFIG.location_dropout_prob,
        CONFIG.weight_decay,
        CONFIG.lr_scheduler_type,
    )

    for epoch in range(1, CONFIG.num_epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, criterion, optimizer, device, CONFIG.feature_mode, train=True)
        if val_loader is not None:
            val_loss, val_accuracy = _run_epoch(model, val_loader, criterion, optimizer, device, CONFIG.feature_mode, train=False)
        else:
            val_loss, val_accuracy = 0.0, 0.0
        if scheduler is not None:
            if CONFIG.lr_scheduler_type == "plateau":
                scheduler.step(val_loss if val_loader is not None else train_loss)
            else:
                scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])

        LOGGER.info(
            "Epoch %s/%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f lr=%.6g",
            epoch,
            CONFIG.num_epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            current_lr,
        )

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": len(label_to_index),
            "label_to_index": label_to_index,
            "index_to_label": index_to_label,
            "feature_mode": CONFIG.feature_mode,
            "feature_spec": feature_spec,
            "skeleton_dim": slices.skeleton_dim,
            "location_dim": slices.location_dim,
            "motion_dim": slices.motion_dim,
            "skeleton_branch_hidden_dim": CONFIG.skeleton_branch_hidden_dim,
            "location_branch_hidden_dim": CONFIG.location_branch_hidden_dim,
            "motion_branch_hidden_dim": CONFIG.motion_branch_hidden_dim,
            "fusion_hidden_dim": CONFIG.fusion_hidden_dim,
            "dropout": CONFIG.multibranch_dropout,
            "bidirectional": CONFIG.multibranch_bidirectional,
            "skeleton_branch_type": CONFIG.skeleton_branch_type,
            "use_gcn_skeleton": CONFIG.use_gcn_skeleton,
            "gcn_hidden_dims": list(CONFIG.gcn_hidden_dims),
            "stgcn_hidden_dim": CONFIG.stgcn_hidden_dim,
            "stgcn_num_blocks": CONFIG.stgcn_num_blocks,
            "stgcn_block_channels": list(CONFIG.stgcn_block_channels),
            "stgcn_dropout": CONFIG.stgcn_dropout,
            "stgcn_use_residual": CONFIG.stgcn_use_residual,
            "stgcn_temporal_kernel_size": CONFIG.stgcn_temporal_kernel_size,
            "use_spatial_translation_aug": CONFIG.use_spatial_translation_aug,
            "spatial_translation_min_offset": CONFIG.spatial_translation_min_offset,
            "spatial_translation_max_offset": CONFIG.spatial_translation_max_offset,
            "location_dropout_prob": CONFIG.location_dropout_prob,
            "use_weighted_loss": CONFIG.use_weighted_loss,
            "weight_decay": CONFIG.weight_decay,
            "lr_scheduler_type": CONFIG.lr_scheduler_type,
            "lr_scheduler_min_lr": CONFIG.lr_scheduler_min_lr,
            "lr_scheduler_patience": CONFIG.lr_scheduler_patience,
            "lr_scheduler_factor": CONFIG.lr_scheduler_factor,
            "current_lr": current_lr,
            "fusion_input_dim": model.fusion_input_dim,
            "epoch": epoch,
        }
        torch.save(checkpoint_payload, last_checkpoint_path)

        if val_loader is None or val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(checkpoint_payload, best_checkpoint_path)

    with (CONFIG.models_dir / "multibranch_train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "data_dir": str(resolved_data_dir),
                "feature_mode": CONFIG.feature_mode,
                "feature_dim": input_dim,
                "skeleton_dim": slices.skeleton_dim,
                "location_dim": slices.location_dim,
                "motion_dim": slices.motion_dim,
                "fusion_input_dim": model.fusion_input_dim,
                "skeleton_branch_type": CONFIG.skeleton_branch_type,
                "use_gcn_skeleton": CONFIG.use_gcn_skeleton,
                "gcn_hidden_dims": list(CONFIG.gcn_hidden_dims),
                "stgcn_hidden_dim": CONFIG.stgcn_hidden_dim,
                "stgcn_num_blocks": CONFIG.stgcn_num_blocks,
                "stgcn_block_channels": list(CONFIG.stgcn_block_channels),
                "stgcn_dropout": CONFIG.stgcn_dropout,
                "stgcn_use_residual": CONFIG.stgcn_use_residual,
                "stgcn_temporal_kernel_size": CONFIG.stgcn_temporal_kernel_size,
                "use_spatial_translation_aug": CONFIG.use_spatial_translation_aug,
                "spatial_translation_min_offset": CONFIG.spatial_translation_min_offset,
                "spatial_translation_max_offset": CONFIG.spatial_translation_max_offset,
                "location_dropout_prob": CONFIG.location_dropout_prob,
                "use_weighted_loss": CONFIG.use_weighted_loss,
                "weight_decay": CONFIG.weight_decay,
                "lr_scheduler_type": CONFIG.lr_scheduler_type,
                "lr_scheduler_min_lr": CONFIG.lr_scheduler_min_lr,
                "lr_scheduler_patience": CONFIG.lr_scheduler_patience,
                "lr_scheduler_factor": CONFIG.lr_scheduler_factor,
                "final_lr": float(optimizer.param_groups[0]["lr"]),
                "best_checkpoint": str(best_checkpoint_path),
                "last_checkpoint": str(last_checkpoint_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    LOGGER.info("Saved best checkpoint to %s", best_checkpoint_path)
