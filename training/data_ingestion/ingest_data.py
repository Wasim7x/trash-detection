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
from utils.custom_transform import get_training_transforms, get_test_val_transforms, TransformDataset

class DataIngestor:
    def __init__(self, config_file: str):
        self.confg = ConfigReader.get_config(config_file)
        self.verbose = self.confg.get("verbose", False)
        self.train_ratio = self.confg.get("train_ratio", 0.1)
        self.val_ratio = self.confg.get("val_ratio", 0.05)
        self.test_ratio = 1 - (self.train_ratio + self.val_ratio)
        self.seed = self.confg.get("seed", 42)
        self.batch_size = self.confg.get("batch_size", 32)
        self.num_workers= self.confg.get("num_workers",4)
        self.dataset_path = Path(Path(__file__).resolve().parent.parent.parent,"garbage-dataset") 
        if not self.dataset_path:
            raise ValueError("Output path is not specified in the configuration file.")
        
        self.full_dataset = self._load_dataset()
        self.class_names = self.full_dataset.classes
        self.num_classes = len(self.class_names)


    def _load_dataset(self):
        """Loads dataset from directory using ImageFolder."""
        print("dataset_path_is : ", self.dataset_path)

        return datasets.ImageFolder(root=self.dataset_path)
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Splits the dataset and returns training and validation DataLoaders."""
        print(len(self.full_dataset))
        torch.manual_seed(self.seed)
        train_data, test_data, val_data= random_split(self.full_dataset, [self.train_ratio, self.test_ratio, self.val_ratio])

        train_dataset = TransformDataset(train_data, get_training_transforms())
        test_dataset = TransformDataset(test_data, get_test_val_transforms())
        val_dataset = TransformDataset(val_data, get_test_val_transforms())
        print("dataset after transformation: ", val_dataset)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        print("data loader successful", train_loader)

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
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    
if __name__ == "__main__":
    main()
    