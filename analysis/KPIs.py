import numpy as np
import pandas as pd


def compute_theils_l(distribution: np.array):
    epsilon = 1e-6
    distribution = distribution + epsilon
    mean = np.mean(distribution)
    theils_l_index = np.mean(np.log(mean / distribution))
    return theils_l_index


def theils_l_group(distribution: np.array, n: int):
    #epsilon = 1e-6
    #distribution = distribution + epsilon
    mean = np.mean(distribution)
    theils_l = (1 / n) * np.sum(np.log(mean / distribution))
    return theils_l


def theils_l_within_and_between(full_distribution: np.array,
                                group_wise_distributions: list[np.array]):
    # avoid division by zero
    epsilon = 1e-6
    full_distribution = full_distribution + epsilon
    group_wise_distributions = [distribution + epsilon for distribution in group_wise_distributions]

    # within component of Theil's L
    theils_l_within = np.array(
        [theils_l_group(distribution, len(full_distribution)) for distribution in group_wise_distributions])

    # between component of Theil's L
    group_wise_means = np.array([np.mean(distribution) for distribution in group_wise_distributions])
    group_sizes = np.array([len(distribution) for distribution in group_wise_distributions])
    mean = np.mean(full_distribution)
    theils_l_between = (1 / len(full_distribution)) * np.sum(group_sizes * np.log(mean / group_wise_means))

    return theils_l_within, theils_l_between


def transform_theil_to_atkinson(value: float):
    if pd.notna(value):
        return 1 - np.exp(-value)
    else:
        return value




def compute_manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))


def compute_euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def min_max_normalize(result: np.array):
    min_value = np.min(result)
    max_value = np.max(result)
    result = np.where(
        min_value != max_value,
        ((result - min_value) / (max_value - min_value)),
        result
    )
    return result
