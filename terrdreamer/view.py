import tempfile
from typing import Tuple

import numpy as np
from mayavi import mlab
from PIL import Image
from scipy.ndimage import gaussian_filter
from tvtk.api import tvtk


def smooth_heightmap(heightmap: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    smoothed_heightmap = gaussian_filter(heightmap, sigma=sigma)
    return smoothed_heightmap


def draw_geo_surface(
    heightmap: np.ndarray,
    texture: np.ndarray,
    size: Tuple[int, int] = (1000, 1000),
    bgcolor: Tuple[float, float, float] = (1, 1, 1),
    show_sealevel: bool = True,
    camera_position: Tuple[float, float, float] = (0, 0, 0),
):
    # Smooth the heightmap
    smoothed_heightmap = smooth_heightmap(heightmap)

    assert list(smoothed_heightmap.shape[:2]) == list(
        texture.shape[:2]
    ), "heightmap and texture must have the same shape"

    # Create x, y coordinates as a meshgrid
    x, y = np.mgrid[: heightmap.shape[0], : heightmap.shape[1]]

    # Set heightmap for a better 3D visualization effect
    z = smoothed_heightmap.astype(np.float32)

    # Create a temporary JPG to hold the texture
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        Image.fromarray(texture).save(f.name)
        bmp1 = tvtk.JPEGReader()
        bmp1.file_name = f.name

        # Instantiate the texture object
        tex_object = tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)

        # Create a figure
        mlab.figure(size=size, bgcolor=bgcolor)

        if show_sealevel:
            mlab.surf(
                x, y, np.zeros_like(z), color=(0, 0, 1), warp_scale="auto", opacity=0.5
            )

        surf = mlab.surf(x, y, z, color=(1, 1, 1), warp_scale="auto")
        surf.actor.enable_texture = True
        surf.actor.tcoord_generator_mode = "plane"
        surf.actor.actor.texture = tex_object

        # Set camera position
        mlab.view(*camera_position)

        # Update the scene
        mlab.draw()
        mlab.show()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from terrdreamer.dataset import AW3D30Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, default="train_aw3d30")
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()

    dataset = AW3D30Dataset(args.dataset_path, normalize=False)

    if args.index is None:
        index = np.random.randint(0, len(dataset))
    else:
        index = args.index

    # Get the image and the depth map
    image, depth = dataset[index]

    # Make sure the image is in the correct format for numpy
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    depth = depth.squeeze().numpy()

    # Save the image and the depth map both as images
    Image.fromarray(image).save(f"index_{index}.jpg")

    # Normalize the depth map
    normed_depth = (depth - depth.min()) / (depth.max() - depth.min())
    normed_depth = (normed_depth * 255).astype(np.uint8)
    Image.fromarray(normed_depth).save(f"index_{index}_depth.jpg")

    draw_geo_surface(depth, image, show_sealevel=False, camera_position=(45, 45, 1000))
