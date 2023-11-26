from typing import List, Sequence, Tuple

import torch

from torch.utils.data import Dataset, Subset, random_split


class SystemIdDataset(Dataset):
    def __init__(
        self, q: torch.Tensor, v: torch.Tensor, vd: torch.Tensor, tau: torch.Tensor
    ):
        """
        Args:
            q (torch.Tensor): The joint positions of shape (num_samples, num_joints).
            v (torch.Tensor): The joint velocities of shape (num_samples, num_joints).
            vd (torch.Tensor): The joint accelerations of shape (num_samples,
            num_joints).
            tau (torch.Tensor): The measured joint torques of shape (num_samples,
            num_joints).
        """
        self._q = q
        self._v = v
        self._vd = vd
        self._tau = tau

    def __len__(self):
        return len(self._q)

    def __getitem__(self, index):
        return {
            "q": self._q[index],
            "v": self._v[index],
            "vd": self._vd[index],
            "tau": self._tau[index],
        }


def sequential_split(dataset: Dataset, lengths: Sequence[int]) -> List[Subset]:
    """Splits a dataset into subsets of sequential indices. No random shuffling is
    performed.

    Args:
        dataset (Dataset): The dataset to split.
        lengths (Sequence[int]): The lengths of each subset.

    Returns:
        List[Subset]: A list of subsets.
    """
    assert sum(lengths) == len(dataset)
    indices = list(range(len(dataset)))
    subsets = []
    for length in lengths:
        subset_indices = indices[:length]
        indices = indices[length:]
        subsets.append(Subset(dataset, subset_indices))
    return subsets


def split_dataset_into_train_val_test(
    dataset: Dataset, train_ratio: float, val_ratio: float, shuffle: bool
) -> Tuple[Dataset, Dataset, Dataset]:
    """Splits a dataset into train, validation, and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): The ratio of the dataset to use for training in range
        [0, 1].
        val_ratio (float): The ratio of the dataset to use for validation in range
        [0, 1].
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: A tuple of (train_dataset, val_dataset,
        test_dataset)
    """
    assert train_ratio + val_ratio < 1.0
    num_train = int(train_ratio * len(dataset))
    num_val = int(val_ratio * len(dataset))
    num_test = len(dataset) - num_train - num_val
    train_dataset, val_dataset, test_dataset = (
        random_split(dataset, [num_train, num_val, num_test])
        if shuffle
        else sequential_split(dataset, [num_train, num_val, num_test])
    )
    print(f"Num train: {num_train}, Num val: {num_val}, Num test: {num_test}")
    return train_dataset, val_dataset, test_dataset
