
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Dict
import lightning as L
from abc import ABC, abstractmethod
import numpy as np


class CustomDataModule(L.LightningDataModule, ABC):

    def __init__(self, configs: dict) -> None:
        """Initialize a custom data module.

        Args:
            configs (dict): Dictionary containing all the configuration. For this class it needs configs["LighningDataModule"] = {TODO}
        """
        super().__init__()
        self.configs = configs
        self.usecase: Literal["target", "general_distribution"] = "target"
        self.datamodule_configs = configs["LightningDataModule"]


    @abstractmethod
    def setup(self, stage: str) -> None:
        # load all the file_paths
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def population_dataloader(self) -> DataLoader:
        pass

    # @property
    # @abstractmethod
    # def population(self) -> Dataset:
    #     pass

    # @property
    # @abstractmethod
    # def train_test_data(self) -> Dict[str, np.ndarray]:
    #     pass
