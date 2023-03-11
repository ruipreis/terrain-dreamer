import tempfile
import urllib
import ee
import ray

import zipfile
from PIL import Image


def initialize_google_earth():
    # If the auth mode isnt provided, browser mode is used by default.
    # then rune initialzie without arguments.

    # If initialize fails, run the following commented line without script,
    # to setup the authentication.
    # ee.Authenticate()

    ee.Initialize()


import rasterio
import rasterio.features
import rasterio.warp
from pathlib import Path
from osgeo import gdal


def slice_gtif(file: Path, out_folder: Path, batch_size: int = 512):
    # Slices a geotiff file into smaller files of batch_size x batch_size
    # and saves them in out_folder. The output files are named as
    # <stem>_<x>_<y>.tif, where <x> and <y> are the index of the batch
    # in the x and y direction respectively.
    stem = file.stem
    gtif = gdal.Open(str(file))
    band = gtif.GetRasterBand(1)

    for i, base_x in enumerate(range(0, band.XSize, batch_size)):
        for j, base_y in enumerate(range(0, band.YSize, batch_size)):
            out_path = out_folder / f"{stem}_{i}_{j}.tif"

            gdal.Translate(
                str(out_path), gtif, srcWin=[base_x, base_y, batch_size, batch_size]
            )


def process_gtif_dataset(dataset_path: Path, out_folder: Path, batch_size: int = 512):
    gtif_files = list(dataset_path.glob("*.tif"))

    # Create a function that runs slice gTIF on a list of files
    @ray.remote
    def _process_gtif_dataset(gtif_files, out_folder, batch_size):
        for file in gtif_files:
            slice_gtif(file, out_folder, batch_size)

    # Slice the files in batches of 100
    slice_refs = [
        _process_gtif_dataset.remote(gtif_files[i : i + 100], out_folder, batch_size)
        for i in range(0, len(gtif_files), 100)
    ]

    # Wait for the slicing to complete
    ray.get(slice_refs)


def gtif_to_geojson(file: Path):
    with rasterio.open(file) as src:
        # Get the dimensions
        height, width = src.shape

        shapes = rasterio.features.shapes(src.dataset_mask(), transform=src.transform)

    out_shapes = []

    for geom, _ in shapes:
        # p_geom = rasterio.warp.transform_geom(src.crs, "EPSG:4326", geom, precision=6)

        out_shapes.append(geom)

    return [ee.Geometry(s).toGeoJSON() for s in out_shapes][0], height, width


def gtif_to_satelite(gtif_file: str, out_file: str, scale: int = 25):
    # This function takes a geotiff file and returns a satelite image
    # with the same dimensions as the geotiff file
    geom, height, width = gtif_to_geojson(gtif_file)

    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(geom)
        .select(["B4", "B3", "B2"])
        .filter(ee.Filter.calendarRange(2022, 2023, "year"))
        .filter(ee.Filter.calendarRange(10, 12, "month"))
    )

    image = dataset.reduce("median")
    percentiles = image.reduceRegion(
        ee.Reducer.percentile([0, 100], ["min", "max"]), geom, scale, bestEffort=True
    ).getInfo()

    # Extracting the results is annoying because EE prepends the channel name
    mymin = [
        percentiles["B4_median_min"],
        percentiles["B3_median_min"],
        percentiles["B2_median_min"],
    ]
    mymax = [
        percentiles["B4_median_max"],
        percentiles["B3_median_max"],
        percentiles["B2_median_max"],
    ]

    minn = max(mymin)
    maxx = max(mymax)

    NEWRGB = ["B4_median", "B3_median", "B2_median"]
    reduction = image.visualize(
        bands=NEWRGB, min=[minn, minn, minn], max=[maxx, maxx, maxx], gamma=1
    )

    path = reduction.getDownloadUrl(
        {
            "scale": scale,
            "crs": "EPSG:4326",
            "maxPixels": 1e20,
            "region": geom,
            "bestEffort": True,
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".zip") as f:
        with tempfile.TemporaryDirectory() as tmpdir:
            urllib.request.urlretrieve(path, f.name)

            with zipfile.ZipFile(f.name, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # This extract 3 tif files to the temporary directory, which
            # are download.vis-(blue, green, red).tif, we need to join them into
            # a single tif file
            red = Image.open(tmpdir + "/download.vis-red.tif")
            green = Image.open(tmpdir + "/download.vis-green.tif")
            blue = Image.open(tmpdir + "/download.vis-blue.tif")

            rgb = Image.merge("RGB", (red, green, blue))
            rgb = rgb.resize((int(width), int(height)))

            rgb.save(out_file)
