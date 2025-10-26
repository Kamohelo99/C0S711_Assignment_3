# Radio Source Classification Pipeline

This repository provides a **skeleton codebase** for building a two‑phase deep learning pipeline to classify radio sources in the MeerKAT MGCLS survey.  It is **not a complete solution**; instead it outlines the structure and key components you need to implement yourself.

The project is organised into two phases:

1. **Supervised training** on the small set of labelled images (typical and exotic sources) using a convolutional neural network (CNN).  You will need to parse image filenames to extract their coordinates, match them to the closest entry in `labels.csv`, build a dataset class, and train a ResNet‑18 backbone from scratch.

2. **Semi‑supervised learning** to make use of the much larger unlabelled dataset.  This stage involves generating pseudo‑labels for unlabelled images using the supervised model, then fine‑tuning the model via consistency regularisation (e.g. FixMatch) or another semi‑supervised strategy.  This skeleton does not implement FixMatch, but provides placeholders for the core logic.

Please read through the code in the `src/` directory and the notebook templates in `notebooks/`.  You will need to fill in the `TODO` sections with your own implementations for parsing coordinates, matching labels, training loops, semi‑supervised learning, and evaluation.  We have included comments and high‑level guidance throughout the files to help you get started.

## Structure

```
radio_source_pipeline/
├── README.md          # This file
├── requirements.txt   # Suggested Python dependencies
├── src/               # Source code (datasets, models, training scripts)
│   ├── dataset.py     # Dataset class and label matching helper functions
│   ├── model.py       # Model definition (ResNet‑based classifier)
│   ├── train_supervised.py   # Skeleton script for supervised training
│   ├── train_semi_supervised.py  # Skeleton script for semi‑supervised training
│   └── utils.py       # Utility functions (metrics, logging)
└── notebooks/
    ├── supervised_template.ipynb      # Jupyter notebook skeleton for supervised training
    └── semi_supervised_template.ipynb # Jupyter notebook skeleton for semi‑supervised training
```

## Quick Start

1. **Install dependencies**.  Create a Python environment (e.g. via `conda` or `virtualenv`) and run:

   ```bash
   pip install -r requirements.txt
   ```

   You may need to adjust versions to suit your hardware and environment (e.g. GPU support for PyTorch).

2. **Prepare the data**.  Extract the ZIP archives for typical (`typ.zip`), exotic (`exo.zip`), and unlabelled (`unl.zip`) sources into separate folders.  Place the `labels.csv` file in a known location.  The dataset class in `src/dataset.py` includes placeholders for loading images and matching them to labels based on their coordinates.

3. **Run the supervised training notebook**.  Open `notebooks/supervised_template.ipynb` in Jupyter or Google Colab.  Follow the instructions in each cell to load the data, implement the dataset parsing and model definition, and complete the training loop.  Save your model checkpoint (e.g. `resnet18_supervised.pth`) for use in the semi‑supervised phase.

4. **Run the semi‑supervised training notebook**.  Open `notebooks/semi_supervised_template.ipynb`.  This notebook guides you through generating pseudo‑labels for unlabelled images and implementing a consistency‑based fine‑tuning loop.  You will need to adapt the pseudo‑labelling threshold and augmentation functions for your data.

## Disclaimer

The provided code is **incomplete** and intended for educational purposes.  It does not include a full implementation of the assignment and will not, by itself, produce a trained model.  You must fill in the missing pieces according to your needs and the assignment requirements.
