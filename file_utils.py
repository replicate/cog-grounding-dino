import subprocess
import tarfile
import os


""" Download the tar file, extract the weights, and move them to the "dest" directory. """


def download_weights(url, dest):
    if not os.path.exists("/src/tmp.tar"):
        print(f"Downloading {url}...")
        try:
            output = subprocess.check_output(["pget", url, "/src/tmp.tar"])
        except subprocess.CalledProcessError as e:
            # If download fails, clean up and re-raise exception
            raise e
    tar = tarfile.open("/src/tmp.tar")
    tar.extractall(path="/src/tmp")
    tar.close()
    os.rename("/src/tmp", dest)
    os.remove("/src/tmp.tar")


WEIGHTS_URL_DIR_MAP = {
    "GROUNDING_DINO_WEIGHTS_URL": "https://weights.replicate.delivery/default/grounding-dino/grounding-dino.tar",
    "HF-CACHE": "https://weights.replicate.delivery/default/grounding-dino/bert-base-uncased.tar",
}


def download_grounding_dino_weights(grounding_dino_weights_dir, hf_cache_dir):
    if not os.path.exists(grounding_dino_weights_dir):
        download_weights(
            url=WEIGHTS_URL_DIR_MAP["GROUNDING_DINO_WEIGHTS_URL"],
            dest=grounding_dino_weights_dir,
        )
    if not os.path.exists(hf_cache_dir):
        download_weights(url=WEIGHTS_URL_DIR_MAP["HF-CACHE"], dest=hf_cache_dir)
