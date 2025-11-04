"""
dataset.py
===========

This module provides classes and functions to load MGCLS images and
associate them with their labels.  The dataset is split into three
parts: `typical` sources, `exotic` sources and `unlabelled` sources.
Only the typical and exotic sets have ground truth labels provided in
`labels.csv`.  Filenames encode sky coordinates that must be matched to
the nearest row in the label CSV.  See the assignment instructions
for details.

This file contains skeleton code only.  You must implement the
`parse_coords_from_filename` and `match_labels` functions as well as
complete the `RadioDataset` class for your use case.
"""

import os
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def parse_coords_from_filename(filename: str) -> Tuple[float, float]:
    """Extract the right ascension and declination coordinates from a filename.

    Filenames in the MGCLS dataset begin with two numbers separated by
    underscores or other characters, for example `123.456_-78.90.png`.
    You need to parse these two numbers and convert them to floats.  In
    some cases the coordinate format may differ, so ensure your parser
    handles variations gracefully.

    Args:
        filename: The name of the image file (without directory).

    Returns:
        A tuple `(ra, dec)` containing the right ascension and declination
        as floats.

    Note:
        This function is intentionally left as a TODO.  You should write
        code here to parse the filename according to the conventions used
        in the dataset.  Feel free to adjust or extend this to suit your
        naming scheme.
    """
    # TODO: implement coordinate parsing
    raise NotImplementedError("parse_coords_from_filename() must be implemented by the user.")


def match_labels(coords: Tuple[float, float], labels_df: pd.DataFrame) -> List[str]:
    """Match a pair of coordinates to the closest label entry in `labels.csv`.

    The assignment specification states that the label coordinates may
    differ slightly from the image coordinates.  You must therefore find
    the nearest row in `labels_df` according to some distance metric
    (e.g. Euclidean distance in RA/Dec space).  Once you find the
    nearest row, return the list of label strings associated with that
    row.  Each row in the CSV may contain multiple label columns
    (e.g. `label1`, `label2`, etc.).

    Args:
        coords: The `(ra, dec)` tuple extracted from the filename.
        labels_df: A pandas DataFrame containing coordinate columns (e.g.
            `'ra'`, `'dec'`) and one or more label columns.

    Returns:
        A list of label strings associated with the nearest coordinate.

    Note:
        This function is intentionally left as a TODO.  You must write
        code here to compute the nearest neighbour and extract labels.
    """
    # TODO: implement nearest neighbour matching
    raise NotImplementedError("match_labels() must be implemented by the user.")


class RadioDataset(Dataset):
    """PyTorch Dataset for the MGCLS radio source images.

    This dataset class loads images from a given directory and matches
    them to labels using the helper functions above.  Images are loaded
    lazily, and optional transforms can be applied.
    """

    def __init__(self,
                 image_root: str,
                 labels_csv: Optional[str] = None,
                 transform: Optional[Any] = None,
                 label_df: Optional[pd.DataFrame] = None) -> None:
        """Initialise the dataset.

        Args:
            image_root: Path to the directory containing image files.
            labels_csv: Path to the CSV file containing labels.  If
                provided, this will be loaded into a DataFrame.  You can
                instead supply an already loaded DataFrame via `label_df`.
            transform: Optional torchvision transform (or callable) to
                apply to the PIL image before returning it.
            label_df: Optional pandas DataFrame of labels.  If both
                `labels_csv` and `label_df` are provided, `label_df`
                takes precedence.
        """
        super().__init__()

        self.image_root = image_root
        self.transform = transform

        # Load the labels DataFrame if provided
        if label_df is not None:
            self.labels_df = label_df
        elif labels_csv is not None:
            self.labels_df = pd.read_csv(labels_csv)
        else:
            self.labels_df = None

        # Enumerate image files in the directory
        self.image_files = []
        for fname in os.listdir(image_root):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.fits')):
                self.image_files.append(fname)
        self.image_files.sort()

        # Precompute labels (optional).  You may choose to compute
        # coordinate matches here and store them, or compute on the fly in
        # __getitem__.  For large datasets this can save time.
        self.labels = {}
        if self.labels_df is not None:
            for fname in self.image_files:
                coords = parse_coords_from_filename(fname)
                try:
                    label_list = match_labels(coords, self.labels_df)
                except NotImplementedError:
                    label_list = []
                self.labels[fname] = label_list

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load an image and its labels.

        Args:
            idx: Index of the item to load.

        Returns:
            A dictionary with keys:
                - `'image'`: a tensor of the image after applying
                  transforms (if any).
                - `'label'`: a list of strings (or a multi-hot vector if
                  you convert labels to indices yourself).
        """
        fname = self.image_files[idx]
        path = os.path.join(self.image_root, fname)
        with Image.open(path) as img:
            # Convert to grayscale if necessary
            img = img.convert('L')
            if self.transform is not None:
                img = self.transform(img)

        # Lookup labels if available; otherwise return an empty list
        labels = self.labels.get(fname, [])

        return {'image': img, 'label': labels, 'filename': fname}
