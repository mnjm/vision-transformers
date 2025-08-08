import tarfile
import requests
import scipy.io
import shutil
from tqdm import tqdm
from pathlib import Path

to_dir = Path("dataset/oxford-flowers102")
base_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
imgs_tgz_name = "102flowers.tgz"
lbls_mat_name = "imagelabels.mat"
split_mat_name = "setid.mat"

def download_file(url:str, dest:Path):
    if not dest.is_file():
        print(f"Downloading {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def extract_tgz(tgz_path, extract_path):
    print(f"Extracting {tgz_path}...")
    with tarfile.open(tgz_path) as tar:
        tar.extractall(path=extract_path, filter='data')

def download_dataset():
    # Download files
    to_dir.mkdir(parents=True, exist_ok=True)
    for name in [imgs_tgz_name, lbls_mat_name, split_mat_name]:
        download_file(base_url + name, to_dir / name)
    # Extract images
    extract_tgz(to_dir / imgs_tgz_name, to_dir)
    # Load labels and split IDs
    labels = scipy.io.loadmat(to_dir / lbls_mat_name)["labels"][0]
    setid = scipy.io.loadmat(to_dir / split_mat_name)
    splits = {
        "train": setid["trnid"][0],
        "val": setid["valid"][0],
        "test": setid["tstid"][0],
    }
    # split and move images
    for split in splits:
        for class_id in range(1, 103):
            (to_dir / split / str(class_id)).mkdir(exist_ok=True, parents=True)
        for idx in splits[split]:
            src_path = to_dir / "jpg" / f"image_{idx:05d}.jpg"
            assert src_path.exists(), f"Image {src_path} doesnt exist"
            lbl = labels[idx-1]
            dest_path = to_dir / split / str(lbl) / src_path.name
            shutil.move(src_path, dest_path)

    # clean up
    shutil.rmtree(to_dir / "jpg")
    for name in [imgs_tgz_name, lbls_mat_name, split_mat_name]:
        (to_dir / name).unlink()

    print("Done")

if __name__ == "__main__":
    download_dataset()