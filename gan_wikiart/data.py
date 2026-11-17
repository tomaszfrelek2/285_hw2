"""WikiArt download + preprocessing. Images are resized once and cached as a uint8 .npz."""
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # some WikiArt scans are huge


def download_wikiart():
    try:  # Colab secret -> env var for kagglehub
        from google.colab import userdata
        os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    except Exception:
        pass
    import kagglehub
    return kagglehub.dataset_download("steubk/wikiart")


def _load(args):
    path, size = args
    try:
        with Image.open(path) as im:
            im.draft("RGB", (4 * size, 4 * size))  # fast JPEG downscaling while decoding
            im = im.convert("RGB")
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            return np.asarray(im.resize((size, size), Image.BICUBIC))
    except Exception:
        return None  # unreadable file


def load_wikiart(cache_file, size=64):
    """Returns (uint8 tensor N x 3 x size x size, list of style names)."""
    if not os.path.exists(cache_file):
        root = download_wikiart()
        paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"preprocessing {len(paths)} images...")
        with Pool(os.cpu_count()) as pool:
            imgs = pool.map(_load, [(str(p), size) for p in paths], chunksize=64)
        keep = [i for i, im in enumerate(imgs) if im is not None]
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez(cache_file, images=np.stack([imgs[i] for i in keep]),
                 labels=np.array([paths[i].parent.name for i in keep]))
    z = np.load(cache_file)
    return torch.from_numpy(z["images"]).permute(0, 3, 1, 2).contiguous(), z["labels"].tolist()
