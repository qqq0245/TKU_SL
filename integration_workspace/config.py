import os
import copy
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    project_root: Path = Path(__file__).resolve().parent
    data_root: Path = field(init=False)
    raw_dir: Path = field(init=False)
    interim_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    exports_dir: Path = field(init=False)
    dataset_pipeline_root: Path = field(init=False)
    raw_videos_dir: Path = field(init=False)
    manifests_dir: Path = field(init=False)
    landmarks_cache_dir: Path = field(init=False)
    processed_sequences_dir: Path = field(init=False)
    splits_dir: Path = field(init=False)
    dataset_logs_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    models_dir: Path = field(init=False)

    camera_index: int = 0
    sequence_length: int = 30
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    feature_mode: str = os.getenv("SIGN_FEATURE_MODE", "landmarks_plus_location_motion")
    use_location_features: bool = True
    use_motion_features: bool = True
    use_zone_encoding: bool = True
    enable_explicit_finger_states: bool = True
    use_spatial_translation_aug: bool = os.getenv("SIGN_USE_SPATIAL_TRANSLATION_AUG", "true").lower() == "true"
    spatial_translation_min_offset: float = float(os.getenv("SIGN_SPATIAL_TRANSLATION_MIN_OFFSET", "0.05"))
    spatial_translation_max_offset: float = float(os.getenv("SIGN_SPATIAL_TRANSLATION_MAX_OFFSET", "0.15"))
    location_reference_strategy: str = "torso_and_face"
    motion_window: int = 2
    export_frame_mask: bool = True
    model_type: str = "lstm"
    multibranch_enabled: bool = True
    skeleton_branch_type: str = os.getenv("SIGN_SKELETON_BRANCH_TYPE", "stgcn")
    use_gcn_skeleton: bool = True
    enable_confidence_threshold: bool = True
    confidence_threshold: float = 0.5
    unknown_label: str = "unknown"
    no_sign_label: str = "no_sign"
    insufficient_signal_label: str = "no_sign"
    enable_temporal_smoothing: bool = True
    smoothing_window: int = 5
    stable_min_count: int = 3
    smoothing_strategy: str = "majority_vote"
    enable_valid_hand_gate: bool = True
    min_valid_frame_ratio: float = 0.4

    left_hand_nodes: int = 21
    right_hand_nodes: int = 21
    pose_indices: tuple[int, ...] = (0, 11, 12, 13, 14, 15, 16, 23, 24)
    channels: int = 3
    explicit_finger_state_dim_per_hand: int = 5
    location_zone_names: tuple[str, ...] = (
        "above_head",
        "face_area",
        "chest_area",
        "left_side",
        "right_side",
        "center_area",
        "lower_torso",
    )
    face_reference_indices: dict[str, int] = field(
        default_factory=lambda: {
            "mouth_upper": 13,
            "mouth_lower": 14,
            "mouth_left": 61,
            "mouth_right": 291,
            "chin": 152,
        }
    )

    hidden_size: int = int(os.getenv("SIGN_HIDDEN_SIZE", "128"))
    num_layers: int = int(os.getenv("SIGN_NUM_LAYERS", "2"))
    dropout: float = float(os.getenv("SIGN_DROPOUT", "0.3"))
    skeleton_branch_hidden_dim: int = int(os.getenv("SIGN_SKELETON_HIDDEN_DIM", "128"))
    location_branch_hidden_dim: int = int(os.getenv("SIGN_LOCATION_HIDDEN_DIM", "64"))
    motion_branch_hidden_dim: int = int(os.getenv("SIGN_MOTION_HIDDEN_DIM", "64"))
    fusion_hidden_dim: int = int(os.getenv("SIGN_FUSION_HIDDEN_DIM", "128"))
    gcn_hidden_dims: tuple[int, ...] = (64, 128)
    stgcn_hidden_dim: int = 128
    stgcn_num_blocks: int = 3
    stgcn_block_channels: tuple[int, ...] = (64, 128, 128)
    stgcn_dropout: float = 0.3
    stgcn_use_residual: bool = True
    stgcn_temporal_kernel_size: int = 9
    multibranch_dropout: float = float(os.getenv("SIGN_MULTIBRANCH_DROPOUT", "0.3"))
    multibranch_bidirectional: bool = os.getenv("SIGN_MULTIBRANCH_BIDIRECTIONAL", "true").lower() == "true"
    location_dropout_prob: float = float(os.getenv("SIGN_LOCATION_DROPOUT_PROB", "0.4"))
    lr_scheduler_type: str = os.getenv("SIGN_LR_SCHEDULER_TYPE", "cosine")
    lr_scheduler_min_lr: float = float(os.getenv("SIGN_LR_SCHEDULER_MIN_LR", "1e-5"))
    lr_scheduler_patience: int = int(os.getenv("SIGN_LR_SCHEDULER_PATIENCE", "4"))
    lr_scheduler_factor: float = float(os.getenv("SIGN_LR_SCHEDULER_FACTOR", "0.5"))
    batch_size: int = int(os.getenv("SIGN_BATCH_SIZE", "16"))
    learning_rate: float = float(os.getenv("SIGN_LEARNING_RATE", "1e-3"))
    weight_decay: float = float(os.getenv("SIGN_WEIGHT_DECAY", "0.0"))
    num_epochs: int = int(os.getenv("SIGN_NUM_EPOCHS", "20"))
    val_ratio: float = 0.2
    random_seed: int = int(os.getenv("SIGN_RANDOM_SEED", "42"))
    dataset_source_mode: str = "default"
    use_weighted_loss: bool = os.getenv("SIGN_USE_WEIGHTED_LOSS", "false").lower() == "true"
    use_weighted_sampler: bool = os.getenv("SIGN_USE_WEIGHTED_SAMPLER", "false").lower() == "true"

    def __post_init__(self) -> None:
        self.data_root = self.project_root / "data"
        self.raw_dir = self.data_root / "raw"
        self.interim_dir = self.data_root / "interim"
        self.processed_dir = self.data_root / "processed"
        self.exports_dir = self.data_root / "exports"
        dataset_pipeline_override = os.getenv("SIGN_DATASET_PIPELINE_ROOT")
        self.dataset_pipeline_root = (
            Path(dataset_pipeline_override)
            if dataset_pipeline_override
            else self.project_root / "dataset_pipeline"
        )
        self.raw_videos_dir = self.dataset_pipeline_root / "raw_videos"
        self.manifests_dir = self.dataset_pipeline_root / "manifests"
        self.landmarks_cache_dir = self.dataset_pipeline_root / "landmarks_cache"
        self.processed_sequences_dir = self.dataset_pipeline_root / "processed_sequences"
        self.splits_dir = self.dataset_pipeline_root / "splits"
        self.dataset_logs_dir = self.dataset_pipeline_root / "logs"
        artifacts_override = os.getenv("SIGN_ARTIFACTS_DIR")
        self.artifacts_dir = Path(artifacts_override) if artifacts_override else self.project_root / "artifacts"
        self.models_dir = self.artifacts_dir / "models"
        self.multibranch_checkpoint_path = self.models_dir / "multibranch_baseline.pt"
        self.project_collection_root = self.project_root.parent
        self.datasets_root = self.project_collection_root / "datasets"
        self.metadata_root = self.project_collection_root / "metadata"
        source_override = os.getenv("SIGN_EXTERNAL_VIDEO_SOURCE_DIR")
        vocab_override = os.getenv("SIGN_VOCABULARY_CSV")
        self.external_video_source_dir = (
            Path(source_override) if source_override else self.datasets_root / "raw" / "00_videos"
        )
        self.vocabulary_list_csv = (
            Path(vocab_override)
            if vocab_override
            else self.metadata_root / "training_30_vocabulary.csv"
        )
        self._apply_root_training_overrides()
        if self.skeleton_branch_type not in {"lstm", "gcn", "stgcn"}:
            raise ValueError(
                f"Unsupported skeleton_branch_type={self.skeleton_branch_type}. "
                "Expected one of: lstm, gcn, stgcn"
            )
        if self.lr_scheduler_type not in {"none", "cosine", "plateau"}:
            raise ValueError(
                f"Unsupported lr_scheduler_type={self.lr_scheduler_type}. "
                "Expected one of: none, cosine, plateau"
            )
        self.use_gcn_skeleton = self.skeleton_branch_type in {"gcn", "stgcn"}
        if self.stgcn_num_blocks != len(self.stgcn_block_channels):
            self.stgcn_num_blocks = len(self.stgcn_block_channels)

    def _apply_root_training_overrides(self) -> None:
        tuning_path = self.project_collection_root / "realtime_tuning_settings.py"
        if not tuning_path.exists():
            return
        try:
            spec = importlib.util.spec_from_file_location("root_realtime_tuning_settings", tuning_path)
            if spec is None or spec.loader is None:
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            overrides = copy.deepcopy(getattr(module, "TRAINING_CONFIG_OVERRIDES", {}))
        except Exception:
            return
        if not isinstance(overrides, dict):
            return
        for key, value in overrides.items():
            if not hasattr(self, key):
                continue
            setattr(self, key, value)


CONFIG = AppConfig()
