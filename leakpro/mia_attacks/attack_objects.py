"""Module for the AttackObjects class."""

import logging
import os
import time

import numpy as np
import torch
import lightning as L

from typing import Type, Optional
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


from leakpro.dataset import Dataset
from leakpro.import_helper import List, Self
from leakpro.model import Model, PytorchModel
from leakpro.models import NN, ConvNet, SmallerSingleLayerConvNet  # noqa: F401
from leakpro.adapters.custom_data_module import CustomDataModule

class AttackObjects:
    """Class representing the attack objects for the MIA attacks."""
    # TODO: Get rid of population and train_test_dataset and just get them from the datamodule
    def __init__(  # noqa: PLR0913
        self,
        data_module: CustomDataModule,
        target_model_class: Type[L.LightningModule],
        trained_target_model: L.LightningModule,
        configs: dict,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize the AttackObjects class.

        Parameters
        ----------
        population : Dataset
            The population.
        train_test_dataset : dict
            The train test dataset.
        target_model : Model
            The target model.
        configs : dict
            The configurations.
        logger : logging.Logger, optional
            The logger, by default None.

        """

        self._data_module = data_module
        self._target_model_class = target_model_class
        self._target_model = trained_target_model

        self._num_shadow_models = configs["audit"]["num_shadow_models"]
        self._configs = configs
        self.logger = logger

        # self._audit_dataset = {
        #     # Assuming train_indices and test_indices are arrays of indices, not the actual data
        #     "data": np.concatenate(
        #         (
        #             train_test_dataset["train_indices"],
        #             train_test_dataset["test_indices"],
        #         )
        #     ),
        #     # in_members will be an array from 0 to the number of training indices - 1
        #     "in_members": np.arange(len(train_test_dataset["train_indices"])),
        #     # out_members will start after the last training index and go up to the number of test indices - 1
        #     "out_members": np.arange(
        #         len(train_test_dataset["train_indices"]),
        #         len(train_test_dataset["train_indices"])
        #         + len(train_test_dataset["test_indices"]),
        #     ),
        # }

        self.log_dir = configs["run"]["log_dir"]

        path_shadow_models = f"{self.log_dir}/shadow_models"

        # Check if the folder does not exist
        if not os.path.exists(path_shadow_models):
            # Create the folder
            os.makedirs(path_shadow_models)

        # List all entries in the directory
        entries = os.listdir(path_shadow_models)
        number_of_files_to_reuse = len(entries)

        # Train shadow models
        if self._num_shadow_models > 0:
            self._shadow_models = []

            for k in range(self._num_shadow_models):

                if number_of_files_to_reuse > 0:
                    shadow_model = self._target_model_class.load_from_checkpoint(f"{path_shadow_models}/model_{k}.ckpt")
                    self._shadow_models.append(shadow_model)
                    number_of_files_to_reuse -= 1
                else:
                    shadow_model = self._target_model_class(configs["LightningModule"]["hparams"])
                    self._data_module.usecase = "general_distribution"
                    # for random subsamples of the dataset change configs["audit"]["f_attack_data_size"]
                    shadow_model = self._target_model_class(configs["LightningModule"])
                    shadow_model = self.train_lightning_shadow_model(shadow_model, self._data_module, dir_shadow_models=path_shadow_models, file_name=f"model_{k}")
                self._shadow_models.append(shadow_model)



    @property
    def shadow_models(self: Self) -> List[L.LightningModule]:
        """Return the shadow models.

        Returns
        -------
        List[L.LightningModule]: The shadow models.

        """
        return self._shadow_models

    @property
    def target_model(self:Self) -> L.LightningModule:
        """Return the target model.

        Returns
        -------
        Model: The target model.

        """
        return self._target_model

    @property
    def audit_dataset(self:Self) -> dict:
        """Return the audit dataset.

        Returns
        -------
            dict: The audit dataset.

        """
        return self._audit_dataset


    def train_lightning_shadow_model(self, shadow_model: L.LightningModule, data_module: L.LightningDataModule, dir_shadow_models: str, file_name: str) -> L.LightningModule:
        """Trains a lightning model with the given data module. Then saves the best model checkpoint and returns the corresponding lightning model.

        Args:
            shadow_model (L.LightningModule): Model object to train.
            data_module (L.LightningDataModule): Data module to provide training and validation data.
            dir_shadow_models (str): Folder where the model will be saved.
            file_name (str): Model name.

        Returns:
            L.LightningModule: Trained model object.
        """
    
        # create callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_shadow_models,
            filename=file_name,
            monitor="val_loss",
        )

        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)

        # instantiate trainer
        trainer = L.Trainer(
            logger=False,
            enable_checkpointing=True,
            accelerator="auto",
            callbacks=[checkpoint_callback, early_stopping_callback],
        )

        trainer.fit(model=shadow_model, datamodule=data_module)
        shadow_model.load_from_checkpoint(checkpoint_callback.best_model_path)
        return shadow_model