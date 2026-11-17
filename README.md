This is the repo for my hw2 code.

Dataset download, model training, and eval was all done on google colab. You should be able to upload this repo too colab and just run the ipynb file.

# DCGAN on WikiArt

- `data.py`: downloads WikiArt from Kaggle and caches it as 64×64 images.
- `models.py`: DCGAN generator and discriminator.
- `train.py`: training loop with checkpointing, sample grids and periodic FID/IS.
- `evaluate.py`: FID and Inception Score using PyTorch-Ignite.
- `plots.py`: training curves, sample grids and nearest-neighbor figure.
- `run_colab.ipynb`: runs everything in Colab.
- `report_figures/`: the figures used in the report.
- `more_data/': contains generated examples, metrics, and logs
## How to run

Upload this folder to Google Drive as `MyDrive/gan_wikiart_code`, add your Kaggle API token as a Colab secret named `KAGGLE_API_TOKEN`, open `run_colab.ipynb` in Colab with a T4 GPU, and Run all.
