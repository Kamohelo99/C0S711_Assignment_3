"""
train_supervised.py
===================

This script provides a skeleton for supervised training of a CNN on
the labelled MGCLS dataset.  It loads the typical and exotic image
sets, matches them to labels in `labels.csv`, constructs a
PyTorch Dataset and DataLoader, builds the model and trains it using
binary cross‑entropy loss.  Many details (data augmentation, class
weights, early stopping, etc.) are left for you to implement.

The goal is to train a model on the small labelled set (around
2,108 images) and save a checkpoint that can be used in the
semi‑supervised stage.

Usage:
    python train_supervised.py --data_root /path/to/data --labels_csv /path/to/labels.csv --output model.pth

You may also run the corresponding notebook in `notebooks/` for an
interactive version of this script.
"""

import argparse
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import RadioDataset
from model import build_model
from utils import compute_f1, compute_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised training for MGCLS radio source classification")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to directory containing typical and exotic images')
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='Path to labels.csv file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Mini‑batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--output', type=str, default='model_supervised.pth',
                        help='Path to save the trained model checkpoint')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of output classes for the model')
    return parser.parse_args()


def create_datasets(data_root: str, labels_csv: str) -> RadioDataset:
    """Create a single dataset by combining typical and exotic images.

    The dataset assumes that `data_root` contains two subdirectories:
    `typ` (typical sources) and `exo` (exotic sources).  Adjust this
    function if your directory structure differs.
    """
    # Combine typical and exotic images by listing both directories
    typ_dir = os.path.join(data_root, 'typ')
    exo_dir = os.path.join(data_root, 'exo')
    # You may wish to read labels once and pass a DataFrame into
    # RadioDataset to avoid reloading on each instantiation.
    label_df = None
    try:
        import pandas as pd
        label_df = pd.read_csv(labels_csv)
    except Exception as e:
        print(f"Warning: failed to read labels CSV: {e}")

    # Define simple data transforms.  You should add more augmentations here.
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    # Initialise separate datasets and combine their file lists.  You can
    # also create a single dataset by merging file lists.  Here we just
    # use one dataset that reads from both directories via a single
    # root; adjust as needed.
    dataset = RadioDataset(image_root=typ_dir, labels_csv=labels_csv, transform=transform, label_df=label_df)
    # Optionally extend dataset.image_files to include exo images.  A
    # simpler approach is to instantiate another RadioDataset and
    # concatenate them via torch.utils.data.ConcatDataset.
    return dataset


def train_epoch(model: torch.nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images = batch['image'].to(device)
        # The labels here are lists of strings.  You need to convert
        # them into multi‑hot vectors before computing loss.  This
        # requires mapping each label string to an integer index.  For
        # simplicity, this skeleton assumes you have preprocessed
        # `batch['label']` into a tensor of shape `(batch_size, num_classes)`
        # elsewhere (perhaps in your Dataset class).  If not, you must
        # implement that yourself.
        labels = batch.get('label_tensor')  # TODO: implement label tensor creation
        if labels is None:
            raise ValueError("Label tensors are missing.  Implement conversion of label lists to multi‑hot vectors.")
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate the model on a validation set and return metrics."""
    model.eval()
    y_true = []
    y_scores = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch.get('label_tensor')
            if labels is None:
                raise ValueError("Label tensors are missing.  Implement conversion of label lists to multi‑hot vectors.")
            outputs = model(images)
            y_true.append(labels.numpy())
            y_scores.append(torch.sigmoid(outputs).cpu().numpy())
    y_true_arr = np.concatenate(y_true, axis=0)
    y_scores_arr = np.concatenate(y_scores, axis=0)
    precision, recall, f1 = compute_f1(y_true_arr, y_scores_arr)
    mAP = compute_map(y_true_arr, y_scores_arr)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'mAP': mAP}


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader
    dataset = create_datasets(args.data_root, args.labels_csv)
    # Split dataset into training and validation sets (e.g. 80/20).  You
    # should stratify by labels if possible.  Here we simply split
    # sequentially as a placeholder.
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    model = build_model(num_classes=args.num_classes, pretrained=False)
    model = model.to(device)

    # Define loss function and optimiser
    criterion = torch.nn.BCEWithLogitsLoss()  # Consider pos_weight for class imbalance
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_f1 = 0.0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={train_loss:.4f}, f1={metrics['f1']:.4f}, mAP={metrics['mAP']:.4f}")
        # Save best model based on F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), args.output)
            print(f"Saved checkpoint to {args.output}")


if __name__ == '__main__':
    main()