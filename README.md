# terrain-dreamer
3D Terrain Generation

## Dataset

The AW3D30 dataset is a global digital elevation model (DEM) produced by the Japan Aerospace Exploration Agency (JAXA). It provides 30-meter resolution digital surface and terrain models for the entire world.

### Downloading the Dataset

The link for downloading the dataset can be found at https://www.eorc.jaxa.jp/ALOS/en/aw3d30/data/index.htm. The user must vary the arguments accordingly. For example, the following link can be used to download the dataset for the area between N075W030 and N080W025: https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2012/N075W030_N080W025.zip .

### Using the Dataset
The AW3D30 dataset can be used for a variety of applications, such as terrain analysis, land cover mapping, and 3D visualization.

### Documentation
For more information about the AW3D30 dataset, please refer to the official documentation at https://www.eorc.jaxa.jp/ALOS/en/aw3d30/aw3d30v3.2_product_e_e1.0.pdf.


### Google Earth

It is assumed user has access to a service account with access to Google Earth API. By default, it is assumed these credentials are placed under `credentials`.

## Requirements

- `apt install libgdal-dev`
    - The `libgdal-dev` package is required for the `gdal` Python package and should have matching versions.
- `requirements.txt`

## Inspiration

The image-to-dem and dem-to-image components of this work take inspiration from the following works by [Emmanouil Panagiotou](https://github.com/Panagiotou), namely:

- **image-to-dem**: [Generating Elevation Surface from a Single RGB Remotely Sensed Image Using Deep Learning][https://github.com/Panagiotou/ImageToDEM].
- **dem-to-image**: [Procedural 3D Terrain Generation using Generative Adversarial Networks](https://github.com/Panagiotou/Procedural3DTerrain).