import os
from typing import Optional, Callable
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    """PyTorch Dataset for GTSRB with optional ROI cropping.

    This dataset supports CSVs containing at least these columns: "Path" and "ClassId".
    If ROI columns are present (Roi.X1, Roi.Y1, Roi.X2, Roi.Y2) the images are cropped
    to the bounding box before any transforms are applied.

    Args:
        csv_file (str): Path to the CSV file (Train.csv).
        root_dir (str): Directory with the images (e.g., data/GTSRB).
        transform (callable, optional): Optional transform to be applied on sample.
        use_roi (bool): If True and ROI columns are present, crop images using ROI.
    """

    def __init__(self, csv_file: str, root_dir: str, transform: Optional[Callable] = None, use_roi: bool = True):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.use_roi = use_roi and set(['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']).issubset(self.df.columns)

        # Normalize path column name
        if 'Path' not in self.df.columns and 'path' in self.df.columns:
            self.df = self.df.rename(columns={'path': 'Path'})

        if 'ClassId' not in self.df.columns and 'classId' in self.df.columns:
            self.df = self.df.rename(columns={'classId': 'ClassId'})

        # Convert to records for faster indexing
        self.samples = self.df.to_dict(orient='records')

    def __len__(self):
        return len(self.samples)

    def _join_path(self, path: str) -> str:
        # CSV path might use forward slashes (Train/...), so split and join to root_dir
        if os.path.isabs(path):
            return path
        parts = list(filter(None, path.replace('\\', '/').split('/')))
        return os.path.join(self.root_dir, *parts)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_rel_path = rec.get('Path')
        if img_rel_path is None:
            raise KeyError("CSV file must contain a 'Path' column pointing to the image file.")

        label = rec.get('ClassId')
        if label is None:
            raise KeyError("CSV file must contain a 'ClassId' column for the label.")

        full_path = self._join_path(img_rel_path)
        image = Image.open(full_path).convert('RGB')

        # Crop to ROI if requested and available
        if self.use_roi:
            try:
                x1 = int(rec['Roi.X1'])
                y1 = int(rec['Roi.Y1'])
                x2 = int(rec['Roi.X2'])
                y2 = int(rec['Roi.Y2'])
                # Ensure valid box
                w, h = image.size
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))
            except Exception:
                # any conversion error, continue and use the full image
                pass

        if self.transform:
            image = self.transform(image)

        return image, int(label)
