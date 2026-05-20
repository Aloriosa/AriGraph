import os
import urllib.request
import tarfile
import shutil

IMAGENET_VAL_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenet1000_val.tar"

def download_and_extract(url, dst):
    tar_path = dst + ".tar"
    print(f"Downloading {url} to {tar_path} ...")
    urllib.request.urlretrieve(url, tar_path)
    print("Extracting...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(dst)
    print("Removing tar file...")
    os.remove(tar_path)

if __name__ == "__main__":
    out_dir = "data/imagenet_val"
    os.makedirs(out_dir, exist_ok=True)
    download_and_extract(IMAGENET_VAL_URL, out_dir)