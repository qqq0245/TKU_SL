from __future__ import annotations

from collections import deque

import numpy as np


class SequenceBuilder:
    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self.frames: deque[np.ndarray] = deque(maxlen=sequence_length)

    def append(self, frame_feature: np.ndarray) -> None:
        self.frames.append(frame_feature.astype(np.float32))

    def reset(self) -> None:
        self.frames.clear()

    def is_full(self) -> bool:
        return len(self.frames) == self.sequence_length

    def __len__(self) -> int:
        return len(self.frames)

    def padded_sequence(self) -> np.ndarray:
        if not self.frames:
            raise ValueError("SequenceBuilder is empty.")
        frames = list(self.frames)
        if len(frames) < self.sequence_length:
            pad_count = self.sequence_length - len(frames)
            padding = [np.zeros_like(frames[0], dtype=np.float32) for _ in range(pad_count)]
            frames = padding + frames
        return np.stack(frames, axis=0).astype(np.float32)

    def sliding_window(self) -> np.ndarray | None:
        if not self.is_full():
            return None
        return np.stack(list(self.frames), axis=0).astype(np.float32)
