import random


def beta_sinalizer(i, delta: float):
    return i * (1 + delta)


def mask_interval(i, delta):
    beta = beta_sinalizer(i, delta)
    return (beta - delta, beta)


def grid_interval(i, delta):
    beta1 = beta_sinalizer(i, delta)
    beta2 = beta_sinalizer(i + 1, delta)
    return (beta1, beta2 - delta)


def inpainting_interval(i, delta):
    beta = beta_sinalizer(i, delta)
    upsilon = beta - delta / 2
    return (upsilon - 0.5, upsilon + 0.5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--inter-real-tile-spacing", type=int, default=3)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--tile-size", type=int, default=256)

    # The number of random inpaintings to perform on each inpainted tile
    parser.add_argument("--n-random-inpaintings", type=int, default=10)

    args = parser.parse_args()

    # Delta corresponds to the percentage of spacing between tiles, as measure
    # of the proportion of the tile size.

    # For example, if delta = 0.5, then the spacing between tiles is 50% of the
    # tile size.

    # Height and width both represent the size of the grid in tiles in that
    # dimension.

    # Thus, depending on the delta, the total size of the grid in pixels will
    # vary.
    delta_adjusted_height = grid_interval(args.height - 1, args.delta)[1]
    delta_adjusted_width = grid_interval(args.width - 1, args.delta)[1]

    # Now, we can to perform inpainting on all of the possible inpaiting intervals
    # in the grid.
    inpainting_intervals = [
        (inpainting_interval(i, args.delta), inpainting_interval(j, args.delta))
        for i in range(args.height)
        for j in range(args.width)
    ]

    # Each inpaint interval corresponds to a tile in the grid. thus, we also want to generate
    # random inpaintings for each tile. A later script will take the box and convert into a
    # valid crop to inpaint in. We only want to inpaint in the regions that are not masked.
    delta_adjustment_constant = 0.5 - args.delta / 2

    random_inpaintings = [
        [
            (
                # Adjust to delta
                random.uniform(
                    vh[0] + delta_adjustment_constant, vh[1] - delta_adjustment_constant
                ),
                random.uniform(
                    vw[0] + delta_adjustment_constant, vw[1] - delta_adjustment_constant
                ),
            )
            for _ in range(args.n_random_inpaintings)
        ]
        for vh, vw in inpainting_intervals
    ]

    print(f"Delta adjusted height: {delta_adjusted_height}")
    print(f"Delta adjusted width: {delta_adjusted_width}")
    import pdb

    pdb.set_trace()
