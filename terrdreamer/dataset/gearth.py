import logging
import tempfile
import urllib
import zipfile
from pathlib import Path

import ee
import rasterio
import rasterio.features
import rasterio.warp
import ray
from osgeo import gdal
from PIL import Image
from ray.util.queue import Queue


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

            if out_path.exists():
                continue

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
        # Get data for the summer months
        # .filter(ee.Filter.calendarRange(6, 8, "month"))
        # Make to sure to filter by could percentage < 5%
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 5))
    )

    reduction = dataset.median()
    percentiles = reduction.reduceRegion(
        ee.Reducer.minMax(), geometry=geom, scale=scale, bestEffort=True
    ).getInfo()

    satImg = reduction.visualize(
        bands=["B4", "B3", "B2"],
        min=[percentiles["B4_min"], percentiles["B3_min"], percentiles["B3_min"]],
        max=[percentiles["B4_max"], percentiles["B3_max"], percentiles["B3_max"]],
        gamma=1,
    )

    path = satImg.getDownloadUrl(
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


def batch_gtif_to_satelite(
    in_folder: Path,
    out_folder: Path,
    scale: int = 25,
    n_consumers: int = 20,
    queue_size: int = 10000,
):
    # The main thread will server as the producer, it will read the files from the
    # input folder and put them in a queue

    # A consumer process that reads data from a queue, initializes google earth engine
    # and then calls gtif_to_satelite on the data, ends when it receives a None value
    @ray.remote
    def _consumer_worker(q: Queue, consumer_id: int):
        logging.info(
            f"Consumer {consumer_id} started, initializing Google Earth Engine ..."
        )
        ee.Initialize()

        while True:
            data = q.get()

            if data is None:
                break

            logging.info(f"Consumer {consumer_id} processing {data}")

            try:
                gtif_to_satelite(*data, scale=scale)
            except Exception as e:
                logging.error(f"Skipping {data} due to processing error: {e}")

    # Create a queue that can hold several files
    q = Queue(maxsize=queue_size)

    # Create a list of consumers
    consumer_workers = [_consumer_worker.remote(q, i) for i in range(n_consumers)]

    for idx, file in enumerate(in_folder.glob("*.tif")):
        out_file = out_folder / (file.stem + ".jpg")

        if out_file.exists():
            logging.info(f"Skipping {file}, output file already exists")
            continue

        logging.info(f"Putting index {idx}, {file} in queue")
        q.put((file, out_file))

    # Put a None value in the queue for each consumer, this will tell the consumer
    # to stop
    for _ in consumer_workers:
        q.put(None)

    logging.info("Waiting for consumers to finish ...")
    ray.get(consumer_workers)

    logging.info("Done!")


if __name__ == "__main__":
    batch_gtif_to_satelite(Path("/freezer/SPLIT_AW3D30"), Path("/freezer/SAT_AW3D30"))
