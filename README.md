![banner](docs/banner.jpg)

Terrain is a novel approach which seeks to solve the procedural terrain generation problem by placing key tiles on a tilemap and then reimagining the missing gaps with inpainting models. This allows us to smoothly transition from one biome to another.

# Requirements

- Git clone the code and install it as a package with `pip install .`
- If you just want to experiment, please download the available checkpoints.
- If you opt for manually downloading and processing the original dataset, it is assumed user has access to a service account with access to Google Earth API

# Dataset

The ALOS World 3D - 30m (AW3D30) is a global Digital Elevation Model (DEM) produced by the Japan Aerospace Exploration Agency (JAXA). It provides 30-meter resolution digital surface and terrain models for the entire world.

## Downloading the Dataset

You may download the dataset from the original source, which is described in the "Preparing the Dataset" section, or alternative download our prepared dataset, which the satelite and depth data already placed in NPZ files and split in training and testing, the link is available [here](https://storage.googleapis.com/terrain-generation-models/AW3D30.zip).

## Preparing the Dataset 

The link for downloading the dataset can be found at https://www.eorc.jaxa.jp/ALOS/en/aw3d30/data/index.htm. The user must vary the arguments accordingly. For example, the following link can be used to download the dataset for the area between N075W030 and N080W025: https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2012/N075W030_N080W025.zip .

Alternative you may use the following script, which automatically downloads the AW3D30 data and obtains the corresponding satelite imagery from google earth API:

1. Download the AW3D30 data.
    ```python
    python terrdreamer/dataset/aw3d30.py
    ```
2. Obtain matching satelite imagery data.
    ```python
    python terrdreamer/dataset/gearth.py
    ```


## Documentation
For more information about the AW3D30 dataset, please refer to the official [documentation](https://www.eorc.jaxa.jp/ALOS/en/aw3d30/aw3d30v3.2_product_e_e1.0.pdf).


## Google Earth

It is assumed user has access to a service account with access to Google Earth API. By default, it is assumed these credentials are placed under `credentials`.

## Requirements

- `apt install libgdal-dev`
    - The `libgdal-dev` package is required for the `gdal` Python package and should have matching versions.
- `requirements.txt`

## Inspiration

The image-to-dem and dem-to-image components of this work take inspiration from the following works by [Emmanouil Panagiotou](https://github.com/Panagiotou), namely:

- **image-to-dem**: [Generating Elevation Surface from a Single RGB Remotely Sensed Image Using Deep Learning](https://github.com/Panagiotou/ImageToDEM).
- **dem-to-image**: [Procedural 3D Terrain Generation using Generative Adversarial Networks](https://github.com/Panagiotou/Procedural3DTerrain).

# Models

We used a total of 4 models to achieve our results, namely:

- **image-to-dem**: We used a [pix2pix](https://arxiv.org/abs/1611.07004) Conditional GAN, that recieves the RGB data as input and outputs a Digital Elevation Model (DEM).
- **dem-to-image**: We also used pix2pix, but now with DEM as input and RGB data as output.
- **image generation**: We used a [ProGAN](https://arxiv.org/abs/1710.10196) to generate novel satelitte imagery.
- **inpainting**: We implemented the inpainting model described in [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892), masking out parts of the salitte data, which we then try to predict.

# Training

All training mechanisms use `wandb`, so please make sure you've installed and properly configured it. In some scripts the `limit` argument can be used to reduce the dataset size for training and experimentation.

## Inpainting model

```bash
python terrdreamer/models/infinity_grid/train.py \
    --train-dataset aw3d30/train \
    --test-dataset aw3d30/test \
    --limit 10000 \
    --save-model-path checkpoints/inpainting \
    --wandb-project inpainting;
```

![inpainting](docs/epoch_200.png)

## Checkpoints

You may download the checkpoints for all of the models via the following command which downloads a zip file containing the checkpoints for all of the models. Please run the following command:

```
mkdir -p checkpoints ; wget https://storage.googleapis.com/terrain-generation-models/checkpoints.zip -P checkpoints ; unzip checkpoints/checkpoints.zip -d checkpoints
```

# Inference

To generate a tile map in our system, we take the following steps:

1. Initilize a qdrant container
    ```
    docker pull qdrant/qdrant
    docker run -p 6333:6333 qdrant/qdrant
    ```
2. Populate the QDrant vector database with randomly generated satelite imagery
    ```python
    python terrdreamer/grid/__init__.py
    ```
3. Create a set of real tiles, which define the base biome, and fake tiles, which will be used for interpolation.
    ```python
    python terrdreamer/grid/generate.py --height 10 --width 50
    ```
4. Do interpolation between previously placed tiles
    ```python
    python terrdreamer/grid/interpolation.py --height 10 --width 50
    ```
5. Fill with inpainting 
    ```python 
    python terrdreamer/grid/filling.py --height 10 --width 50
    ```
6. Do depth estimation for image tiles
    ```python 
    python terrdreamer/grid/depth.py
    ```
7. Convert it back to image format
    ```python 
    python terrdreamer/grid/to_image.py
    ```

# Dataset

The dataset, which corresponds to 18,000 tuples of RGB satelitte images and Digital Surface Models (DSM), each corresponding to different coordinates in the original ALOS World 3D-30m (AW3D30) dataset. You may download this dataset [here](https://storage.googleapis.com/terrain-generation-models/AW3D30.zip). Please make sure to unzip the dataset to make it acessible to the training script.
