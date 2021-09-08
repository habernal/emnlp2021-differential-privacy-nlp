from typing import List

import numpy as np
from numpy.random import default_rng


def ratio_of_points_with_distance_larger_than_two_c(
        pairs_of_points: np.ndarray,
        clipping_constant: float,
        debug=False,
        clip_by_l1_instead=False) -> float:
    """

    :param pairs_of_points:
    :param clipping_constant: 'C' in the paper
    :param debug:
    :param clip_by_l1_instead:
    :return:
    """
    if debug:
        print('pairs_of_points', pairs_of_points)

    l2_norm = np.linalg.norm(pairs_of_points, ord=None, axis=-1, keepdims=True)

    if debug:
        print('l2_norm\n', l2_norm)

    clipped = clipping_constant / l2_norm

    # just as sanity check: clip by l1 norm
    if clip_by_l1_instead:
        l1_norm = np.linalg.norm(pairs_of_points, ord=1, axis=-1, keepdims=True)
        clipped = clipping_constant / l1_norm

    if debug:
        print('clipped\n', clipped)

    ones_like_clipped = np.ones_like(clipped)
    normalizing_constant = np.minimum(ones_like_clipped, clipped)

    if debug:
        print('normalizing_constant\n', normalizing_constant)

    f_y_clipped = pairs_of_points * normalizing_constant

    if debug:
        print('f_y_clipped\n', f_y_clipped)

    first_vectors_from_pairs = f_y_clipped[:, 0, :]
    second_vectors_from_pairs = f_y_clipped[:, 1, :]

    # print('first_vectors_from_pairs\n', first_vectors_from_pairs)
    # print('second_vectors_from_pairs\n', second_vectors_from_pairs)

    differences_between_points = first_vectors_from_pairs - second_vectors_from_pairs

    if debug:
        print('differences_between_points\n', differences_between_points)

    actual_delta_f = np.linalg.norm(differences_between_points, ord=1, axis=1, keepdims=True)

    if debug:
        print('actual_delta_f\n', actual_delta_f)

    larger_than_paper_delta_f = np.where(actual_delta_f > (2 * clipping_constant), 1.0, 0.0)

    if debug:
        print('larger_than_paper_delta_f\n', larger_than_paper_delta_f)

    # return the percentage as a scalar float
    return (sum(larger_than_paper_delta_f) / len(larger_than_paper_delta_f)).item()


def test_ratio() -> None:
    """
    Sanity test for two known points with known clipping constant
    """
    test_points = np.array([
        [
            [2, 2],  # these two points violate the privacy guarantee
            [-2, -2],
        ],
        [
            [-1, 1],  # these two points adhere to the privacy guarantee
            [3, 3]
        ]
    ])
    r = ratio_of_points_with_distance_larger_than_two_c(test_points, 3, True)
    print(r)
    # so 50% of point pairs are non-privatized
    assert r == 0.5


def test_ratio_random() -> None:
    rng = default_rng(1234)

    dimensions = 3

    no_of_random_point_pairs = 5

    clipping_constant_c = 10

    extended_range_for_samples_beyond_clipping_constant = 1  # multiplier

    uniform = rng.uniform(low=-(clipping_constant_c + extended_range_for_samples_beyond_clipping_constant),
                          high=(clipping_constant_c + extended_range_for_samples_beyond_clipping_constant),
                          size=(no_of_random_point_pairs, 2, dimensions))
    gaussian = rng.normal(loc=0.0,
                          scale=0.1 * (clipping_constant_c + extended_range_for_samples_beyond_clipping_constant),
                          size=(no_of_random_point_pairs, 2, dimensions))

    print(ratio_of_points_with_distance_larger_than_two_c(
        uniform, clipping_constant_c, debug=True))
    print(ratio_of_points_with_distance_larger_than_two_c(
        uniform, clipping_constant_c, debug=True, clip_by_l1_instead=True))
    print(ratio_of_points_with_distance_larger_than_two_c(
        gaussian, clipping_constant_c, debug=True))
    print(ratio_of_points_with_distance_larger_than_two_c(
        gaussian, clipping_constant_c, debug=True, clip_by_l1_instead=True))


# now with random and as a function
def main():
    rng = default_rng(123456)

    for dimensions in range(1, 256):
        # print("Dimensions: ", dimensions)

        collect_uniform: List[float] = []
        collect_gaussian: List[float] = []
        collect_gaussian_tight: List[float] = []
        collect_correct: List[float] = []

        for i in range(1):
            no_of_random_point_pairs = 10_000

            clipping_constant_c = 10
            # the larger the actual range, the more likely to be clipped by the l2 norm
            # and thus lay outside the l1 norm and loose privacy
            extended_range_for_samples_beyond_clipping_constant = i

            uniform = rng.uniform(low=-(clipping_constant_c + extended_range_for_samples_beyond_clipping_constant),
                                  high=(clipping_constant_c + extended_range_for_samples_beyond_clipping_constant),
                                  size=(no_of_random_point_pairs, 2, dimensions))
            gaussian = rng.normal(loc=0.0,
                                  scale=0.1 * (
                                              clipping_constant_c + extended_range_for_samples_beyond_clipping_constant),
                                  size=(no_of_random_point_pairs, 2, dimensions))

            gaussian_tight = rng.normal(loc=0.0,
                                        scale=0.01 * (
                                                clipping_constant_c + extended_range_for_samples_beyond_clipping_constant),
                                        size=(no_of_random_point_pairs, 2, dimensions))

            collect_uniform.append(ratio_of_points_with_distance_larger_than_two_c(
                uniform, clipping_constant_c, debug=False))
            collect_gaussian.append(ratio_of_points_with_distance_larger_than_two_c(
                gaussian, clipping_constant_c, debug=False))
            collect_gaussian_tight.append(ratio_of_points_with_distance_larger_than_two_c(
                gaussian_tight, clipping_constant_c, debug=False))

            collect_correct.append(ratio_of_points_with_distance_larger_than_two_c(
                gaussian, clipping_constant_c, debug=False, clip_by_l1_instead=True))

        minimum_uniform = np.array(collect_uniform).min(initial=np.infty)
        mean_uniform = np.array(collect_uniform).mean()

        minimum_gaussian = np.array(collect_gaussian).min(initial=np.infty)
        minimum_gaussian_tight = np.array(collect_gaussian_tight).min(initial=np.infty)
        mean_gaussian = np.array(collect_uniform).mean()
        minimum_correct = np.array(collect_correct).min(initial=np.infty)
        print(dimensions, minimum_uniform, minimum_gaussian, minimum_gaussian_tight, minimum_correct, sep='\t')


if __name__ == '__main__':
    # test_ratio()
    # test_ratio_random()
    main()
