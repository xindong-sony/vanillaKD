import torch
import numpy as np
import random
from collections import defaultdict, Counter


def sample_indices(labels, num_samples, seed=123):
    np.random.seed(seed)
    # get all classes
    all_classes = np.unique(labels)
    class_to_indices = {i: np.where(np.array(labels) == i)[0] for i in all_classes}
    sampled_indices = [np.random.choice(class_indices, num_samples, replace=False) for class_indices in class_to_indices.values()]
    return np.concatenate(sampled_indices).tolist()

def get_targets_from_timm_dataset(dataset):
    return [t for x,t in dataset.parser.samples]


def get_subset(dataset, num_samples, seed=123):
    targets = get_targets_from_timm_dataset(dataset)
    sampled_indices = sample_indices(targets, num_samples, seed)

    dataset.parser.samples = [dataset.parser.samples[i] for i in sampled_indices]

    return dataset

if __name__ == "__main__":
    labels = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    sampled_indices = sample_indices(labels, num_samples=2, seed=123)
    print(sampled_indices)
    print([labels[i] for i in sampled_indices])