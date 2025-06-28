import os
import sys
import torch
from pathlib import Path
sys.path[0] = str(Path(__file__).resolve().parent.parent)
# from data_ingestion.__init__ import *
from utils.config_reader import ConfigReader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, List


class DataIngestor:
    def __init__(self, config_file: str):
        self.confg = ConfigReader.get_config(config_file)
        self.verbose = self.confg.get("verbose", False)
        self.train_ratio = self.confg.get("train_ratio", 0.8)
        self.val_ratio = self.confg.get("val_ratio", 0.1)
        self.test_ratio = self.confg.get("test_ratio", 0.1)
        self.seed = self.confg.get("seed", 42)
        self.batch_size = self.confg.get("batch_size", 32)
        self.num_workers= self.confg.get("num_workers",4)
        self.dataset_path = Path(Path(__file__).resolve().parent.parent.parent,"garbage-dataset") 
        if not self.dataset_path:
            raise ValueError("Output path is not specified in the configuration file.")
        
        self.transform = self._get_transforms()
        self.full_dataset = self._load_dataset()
        self.class_names = self.full_dataset.classes
        self.num_classes = len(self.class_names)


    def _get_trainig_transforms(self):

        """Returns the default trainig transforms for img_classification model like ResNet-50, EfficientNet-B0"""
        
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])                              
        ])
    
    def _get_test_val_transforms(self):

        """Returns the default testing and validation transforms for img_classification model like ResNet-50, EfficientNet-B0"""

        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        ])


    def _load_dataset(self):
        """Loads dataset from directory using ImageFolder."""
        print("dataset_path_is : ", self.dataset_path)

        return datasets.ImageFolder(root=self.dataset_path)
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Splits the dataset and returns training and validation DataLoaders."""

        torch.manual_seed(self.seed)
        train_dataset, test_dataset, val_dataset = random_split(self.full_dataset, [self.train_ratio, self.test_ratio, self.val_ratio])

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        return train_loader, test_loader, val_loader
    
    def get_class_names(self) -> List[str]:
        """Returns class names."""
        return self.class_names

    def get_num_classes(self) -> int:
        """Returns the number of output classes."""
        return self.num_classes


def main():
    config_file = r"training\\configs\\config.yaml"
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} does not exist.")
        sys.exit(1)
    loader = DataIngestor(config_file)
    train_loader, test_loader, val_loader = loader.get_dataloaders()
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.datasets)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    
if __name__ == "__main__":
    main()

    