import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pydicom
import numpy as np
import os
from glob import glob
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OptimizedMRIDataset(Dataset):
    def __init__(self, dicom_files, transform=None, cache=False):
        """
        Optimized dataset for MRI DICOM slices processed as grayscale.
        Args:
            dicom_files (list): List of DICOM file paths.
            transform (callable, optional): Albumentations transform for preprocessing.
            cache (bool): Whether to cache images in memory after loading.
        """
        self.dicom_files = dicom_files
        self.transform = transform
        self.cache = cache
        self.cached_images = {} if cache else None

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        # Check cache
        if self.cache and idx in self.cached_images:
            img = self.cached_images[idx]
        else:
            # Read DICOM file
            dicom_file = self.dicom_files[idx]
            dicom_data = pydicom.dcmread(dicom_file)
            img = dicom_data.pixel_array.astype(np.float32)

            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Ensure 2D grayscale
            if img.ndim > 2:
                img = img[..., 0]

            # Cache if enabled
            if self.cache:
                self.cached_images[idx] = img

        # Apply transforms
        if self.transform:
            img = self.transform(image=img)["image"]

        return img, 0  # Dummy label for Trainer compatibility

class EfficientDataLoader:
    def __init__(self, dataset_path, batch_size=1, num_workers=4, train_split=0.8, val_split=0.1, cache_data=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Efficient data loader for grayscale MRI dataset (256x256).
        Args:
            dataset_path (str): Path to directory with subject folders containing DICOM files.
            batch_size (int): Batch size for data loaders.
            num_workers (int): Number of workers for data loading.
            train_split (float): Proportion of subjects for training.
            val_split (float): Proportion of subjects for validation.
            cache_data (bool): Whether to cache images in memory.
            device (str): Device for data loading ('cuda' or 'cpu').
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.cache_data = cache_data
        self.device = device

        # Define transforms for grayscale 256x256 images
        transform = A.Compose([
            A.Resize(256, 256),  # Resize to 256x256
            A.ToFloat(max_value=1.0),  # Ensure float32
            ToTensorV2(),  # Convert to [1, 256, 256]

        ])

        # Collect subject directories
        subject_dirs = sorted(glob(os.path.join(self.dataset_path, "*")))
        if not subject_dirs:
            raise ValueError(f"No subject folders found in {self.dataset_path}")

        # Subject-aware splitting
        train_subjects, test_subjects = train_test_split(
            subject_dirs, test_size=1.0 - self.train_split, random_state=42
        )
        train_subjects, val_subjects = train_test_split(
            train_subjects, test_size=self.val_split / (self.train_split + self.val_split), random_state=42
        )

        # Collect DICOM files for each split
        train_files = []
        val_files = []
        test_files = []
        for subject_dir in train_subjects:
            train_files.extend(sorted(glob(os.path.join(subject_dir, "*.dcm"))))
        for subject_dir in val_subjects:
            val_files.extend(sorted(glob(os.path.join(subject_dir, "*.dcm"))))
        for subject_dir in test_subjects:
            test_files.extend(sorted(glob(os.path.join(subject_dir, "*.dcm"))))

        if not train_files or not val_files:
            raise ValueError("Empty train or validation set after splitting")

        # Create datasets
        self.train_dataset = OptimizedMRIDataset(train_files, transform, cache=self.cache_data)
        self.eval_dataset = OptimizedMRIDataset(val_files, transform, cache=self.cache_data)
        self.all_dataset = OptimizedMRIDataset(train_files + val_files + test_files, transform, cache=self.cache_data)

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
            persistent_workers=(self.num_workers > 0),
            drop_last=True
        )

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
            persistent_workers=(self.num_workers > 0),
            drop_last=False
        )

        self.all_loader = DataLoader(
            self.all_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
            persistent_workers=(self.num_workers > 0),
            drop_last=True
        )

    def data_scaler(self, x):
        """
        Preprocess data to match VESDE's expected range.
        Args:
            x: Input tensor [batch_size, 1, 256, 256]
        Returns:
            Scaled tensor
        """
        return x  # Input is [0, 1] with 3 repeated channels
        # Alternative: Subtract VESDE prior mean
        # mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, 3, 1, 1)
        # return x - mean

# Register the data loader
_DATA_LOADERS = lambda dataset_path: EfficientDataLoader(dataset_path)