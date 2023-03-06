import os
import requests
from multiprocessing import Pool


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


if __name__ == "__main__":
    import time
    from terrdreamer.dataset.aw3d30 import acquire

    ls = acquire()
    start = time.time()
    download(ls, "AW3D30_DATASET")
    end = time.time()
    print(f"Time taken: {end - start} seconds")
