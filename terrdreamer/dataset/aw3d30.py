import os
import re
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import List

import requests
from tqdm.contrib.concurrent import process_map


def download_file(sample, folder):
    base_name = sample.split("/")[-1]
    print(f"Downloading {base_name}...")

    req_sample = requests.get(sample)

    # If the file is not found or response is HTML, skip
    if (
        req_sample.status_code == 404
        or req_sample.headers["Content-Type"] == "text/html"
    ):
        print(f"File {base_name} not found. Skipping...")
        return

    with open(os.path.join(folder, base_name), "wb") as f:
        f.write(req_sample.content)


def download(ls, folder):
    with Pool() as p:
        p.starmap(download_file, [(sample, folder) for sample in ls])


def _derive_subtiles(src_direction: str, dest_direction: str, degree_step: int):
    src_orientation = src_direction[0]
    dest_orientation = dest_direction[0]
    is_vertical = src_orientation == "N" or src_orientation == "S"

    # Find the degree value of the source and destination
    src_degree = int(src_direction[1:])
    dest_degree = int(dest_direction[1:])

    if src_orientation == "S" or src_orientation == "W":
        src_degree = -src_degree

    if dest_orientation == "S" or dest_orientation == "W":
        dest_degree = -dest_degree

    if src_degree > dest_degree:
        degree_step = -degree_step

    # Now produce the subtiles between the source and destination according to the degree step
    degree_steps = range(src_degree, dest_degree + degree_step, degree_step)

    # Now produce a list of the subtiles, in the format of S010, S005, N000, N005
    basis = ["S", "N"] if is_vertical else ["W", "E"]

    return [
        f"{basis[0] if degree < 0 else basis[1]}{abs(degree):03d}"
        for degree in sorted(degree_steps)
    ]


def acquire() -> List[str]:
    _POSSIBLE_TILES = [
        "N080W030_N090E000",
        "N080E000_N090E030",
        "N080E030_N090E060",
        "N080E060_N090E090",
        "N080E090_N090E120",
        "N080W120_N090W090",
        "N080W090_N090W060",
        "N080W060_N090W030",
        "N050W030_N080E000",
        "N050E000_N080E030",
        "N050E030_N080E060",
        "N050E060_N080E090",
        "N050E090_N080E120",
        "N050E120_N080E150",
        "N050E150_N080E180",
        "N050W180_N080W150",
        "N050W150_N080W120",
        "N050W120_N080W090",
        "N050W090_N080W060",
        "N050W060_N080W030",
        "N020W030_N050E000",
        "N020E000_N050E030",
        "N020E030_N050E060",
        "N020E060_N050E090",
        "N020E090_N050E120",
        "N020E120_N050E150",
        "N020E150_N050E180",
        "N020W180_N050W150",
        "N020W150_N050W120",
        "N020W120_N050W090",
        "N020W090_N050W060",
        "N020W060_N050W030",
        "S010W030_N020E000",
        "S010E000_N020E030",
        "S010E030_N020E060",
        "S010E060_N020E090",
        "S010E090_N020E120",
        "S010E120_N020E150",
        "S010E150_N020E180",
        "S010W180_N020W150",
        "S010W150_N020W120",
        "S010W120_N020W090",
        "S010W090_N020W060",
        "S010W060_N020W030",
        "S040W030_S010E000",
        "S040E000_S010E030",
        "S040E030_S010E060",
        "S040E060_S010E090",
        "S040E090_S010E120",
        "S040E120_S010E150",
        "S040E150_S010E180",
        "S040W180_S010W150",
        "S040W150_S010W120",
        "S040W120_S010W090",
        "S040W090_S010W060",
        "S040W060_S010W030",
        "S070W030_S040E000",
        "S070E000_S040E030",
        "S070E030_S040E060",
        "S070E060_S040E090",
        "S070E090_S040E120",
        "S070E120_S040E150",
        "S070E150_S040E180",
        "S070W180_S040W150",
        "S070W120_S040W090",
        "S070W090_S040W060",
        "S070W060_S040W030",
        "S090W030_S070E000",
        "S090E000_S070E030",
        "S090E030_S070E060",
        "S090E060_S070E090",
        "S090E090_S070E120",
        "S090E120_S070E150",
        "S090E150_S070E180",
        "S090W180_S070W150",
        "S090W150_S070W120",
        "S090W120_S070W090",
        "S090W090_S070W060",
        "S090W060_S070W030",
    ]

    # Keep a record of all of the subtiles
    possible_subtiles = []

    # For each tile in _POSSIBLE_TILES, add the existing sub-tiles to the list
    # subtiles are defined as 5 degree by 5 degree tiles inside the larger tiles
    # For simplicity purposes we represent S as being negative and W as being positive
    # west, W, is negative and east, E, is positive
    for tile in _POSSIBLE_TILES:
        # Extract the lat and lon from the tile name
        matches = re.findall(r"([NS]\d+)([EW]\d+)", tile)

        # Matching latitude and longitude subtiles
        lat_subtiles = _derive_subtiles(matches[0][0], matches[1][0], degree_step=5)
        lon_subtiles = _derive_subtiles(matches[0][1], matches[1][1], degree_step=5)

        # For every subsequent pair, add intervals to the list
        for i in range(0, len(lat_subtiles) - 1):
            for j in range(0, len(lon_subtiles) - 1):
                possible_subtiles.append(
                    f"{lat_subtiles[i]}{lon_subtiles[j]}_{lat_subtiles[i+1]}{lon_subtiles[j+1]}"
                )

    return [
        f"https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2012/{subtile}.zip"
        for subtile in possible_subtiles
    ]


def _extract_dsm_files(x):
    zip_path, out_path = x

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith("_DSM.tif"):
                # Get the indexation of the file name
                dsm_name = file.split("/")[-1].split("_")[1]
                out_file_path = out_path / f"{dsm_name}.tif"

                if out_file_path.exists():
                    continue

                # Extract the file to the output path
                with open(out_file_path, "wb") as f:
                    f.write(zip_ref.read(file))


def extract(path: Path, outpath: Path):
    assert path.exists(), f"Path {path} does not exist"
    assert outpath.exists(), f"Path {outpath} does not exist"

    # Get all of the zip files and extract them with multiprocessing
    zip_files = list(path.glob("*.zip"))

    process_map(
        _extract_dsm_files,
        [(zip_file, outpath) for zip_file in zip_files],
        chunksize=1,
    )


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--outpath", type=Path, required=True)
    args = parser.parse_args()

    if args.extract:
        extract(args.path, args.outpath)
    elif args.download:
        ls = acquire()
        start = time.time()
        download(ls, "AW3D30_DATASET")
        end = time.time()
        print(f"Time taken: {end - start} seconds")
