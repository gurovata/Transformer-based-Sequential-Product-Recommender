from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LeaveTwoOutSplit:
    train: list[int]
    validation: int
    test: int


def leave_two_out(sequence: Sequence[int], min_train_length: int = 1) -> LeaveTwoOutSplit:
    min_length = min_train_length + 2
    if len(sequence) < min_length:
        raise ValueError(
            f"Sequence length must be at least {min_length}, got {len(sequence)}."
        )

    return LeaveTwoOutSplit(
        train=list(sequence[:-2]),
        validation=int(sequence[-2]),
        test=int(sequence[-1]),
    )
