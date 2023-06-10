import numpy as np
from mayavi import mlab
from PIL import Image
from tvtk.api import tvtk
import tempfile
from typing import Tuple

def draw_geo_surface(heightmap:np.ndarray, texture:np.ndarray, size:Tuple[int,int]=(600,500),bgcolor:Tuple[float,float,float]=(1,1,1)):
    assert list(heightmap.shape[:2]) == list(texture.shape[:2]), 'heightmap and texture must have the same shape'

    # Create x, y coordinates as a meshgrid
    x, y = np.mgrid[:heightmap.shape[0], :heightmap.shape[1]]

    # Set heightmap for a better 3D visualization effect
    z = heightmap.astype(np.float32)

    # Create a temporary JPG to hold the texture
    with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
        Image.fromarray(texture).save(f.name)
        bmp1 = tvtk.JPEGReader()
        bmp1.file_name = f.name

        # Instantiate the texture object
        tex_object = tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)

        # Create a figure
        mlab.figure(size=size, bgcolor=bgcolor)

        surf = mlab.surf(x, y, z, color=(1, 1, 1), warp_scale='auto')
        surf.actor.enable_texture = True
        surf.actor.tcoord_generator_mode = 'plane'
        surf.actor.actor.texture = tex_object

        # Update the scene
        mlab.draw()
        mlab.show()