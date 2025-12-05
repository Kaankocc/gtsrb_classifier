import os
from typing import Optional, Callable
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class GTSRBDataset(Dataset):
    """
    A custom PyTorch Dataset for the German Traffic Sign Recognition Benchmark (GTSRB).

    This dataset handles loading images based on a CSV annotation file. It supports:
    1. Automatic handling of 'Path' and 'ClassId' column naming variations.
    2. Optional Region of Interest (ROI) cropping to isolate traffic signs.
    3. On-the-fly transformations.

    Args:
        csv_file (str): Full path to the CSV annotations file (e.g., 'Train.csv').
        root_dir (str): Root directory containing the image subfolders.
        transform (callable, optional): PyTorch transforms to apply (augmentation/normalization).
        use_roi (bool): If True, crops the image to the bounding box specified in the CSV. 
                        Defaults to True.
    """

    def __init__(self, csv_file: str, root_dir: str, transform: Optional[Callable] = None, use_roi: bool = True):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform
        
        # Load the annotations
        self.df = pd.read_csv(csv_file)

        # Check if ROI columns exist to enable cropping
        roi_columns = {'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2'}
        self.use_roi = use_roi and roi_columns.issubset(self.df.columns)

        # Standardize column names to handle variations (e.g., 'path' vs 'Path')
        if 'Path' not in self.df.columns and 'path' in self.df.columns:
            self.df = self.df.rename(columns={'path': 'Path'})

        if 'ClassId' not in self.df.columns and 'classId' in self.df.columns:
            self.df = self.df.rename(columns={'classId': 'ClassId'})

        # Convert DataFrame to a list of dictionaries for faster O(1) indexing during training
        self.samples = self.df.to_dict(orient='records')

    def __len__(self):
        return len(self.samples)

    def _join_path(self, path: str) -> str:
        """
        Constructs the full file path, handling OS separators and relative paths.
        
        Args:
            path (str): The relative path from the CSV (e.g., 'Train/0/img.png').
        
        Returns:
            str: The absolute or correctly joined OS-specific path.
        """
        if os.path.isabs(path):
            return path
        
        # Normalize separators to forward slashes, split, and rejoin using os.path.join
        # to ensure compatibility across Windows/Linux/macOS.
        parts = list(filter(None, path.replace('\\', '/').split('/')))
        return os.path.join(self.root_dir, *parts)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.
        
        Returns:
            tuple: (image_tensor, label) where image_tensor has transformations applied.
        """
        rec = self.samples[idx]
        
        # Validate critical columns
        img_rel_path = rec.get('Path')
        if img_rel_path is None:
            raise KeyError("CSV file must contain a 'Path' column pointing to the image file.")

        label = rec.get('ClassId')
        if label is None:
            raise KeyError("CSV file must contain a 'ClassId' column for the label.")

        # Load image
        full_path = self._join_path(img_rel_path)
        image = Image.open(full_path).convert('RGB')

        # Apply Region of Interest (ROI) cropping if enabled
        if self.use_roi:
            try:
                x1 = int(rec['Roi.X1'])
                y1 = int(rec['Roi.Y1'])
                x2 = int(rec['Roi.X2'])
                y2 = int(rec['Roi.Y2'])
                
                # Defensive Clipping: Ensure coordinates are within actual image bounds.
                # This handles rare cases where CSV annotations might be slightly off.
                w, h = image.size
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                
                # Only crop if the resulting box has valid area
                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))
            except Exception:
                # Fallback: If cropping fails (e.g., bad data), use the full image.
                pass

        # Apply transformations (e.g., Resize, ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)

        return image, int(label)