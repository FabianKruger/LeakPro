from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
from typing import Literal, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from moreno.data_modules.datasets_and_collate_functions import GraphDataset, mol_collate_fn
from leakpro.adapters.custom_data_module import CustomDataModule

class AmesDataModule(CustomDataModule):

    def __init__(self, configs: dict) -> None:

        super().__init__(configs)
        self.train_dataset: Optional[Dataset] = None
        self.validation_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.input_vec_dim: int = 0

        # TODO: check if this can be moved to setup

    def prepare_data(self) -> None:
        # at this point just some error catching. Could technically be removed. Just for easier debugging for now.
        file_paths = self.datamodule_configs["file_paths"]
        for key, file_path in file_paths.items():
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"The {key} file at {file_path} does not exist.")
                   
    def setup(self, stage: str) -> None:
        # load all the file_paths
        if stage == "fit":
            if self.usecase == "target":
                train_data = pd.read_csv(self.datamodule_configs["file_paths"]["train"])
                self.train_dataset = self.convert_dataset(train_data)
                validation_data = pd.read_csv(self.datamodule_configs["file_paths"]["validation"])
                self.validation_dataset = self.convert_dataset(validation_data)
            if self.usecase == "general_distribution":
                population_data = pd.read_csv(self.datamodule_configs["file_paths"]["population"])
                # TODO: ask Johan if this way of taking random subsamples of the auxiliary data makes sense
                # f_shadow_data = self.configs["audit"]["f_attack_data_size"]
                # possible add a seed here for reproducibility
                f_shadow_data = 1
                subset = population_data.sample(frac=f_shadow_data)
                # split into train and validation dataset
                split_point = int(len(subset) * 0.8)
                train_data = subset.iloc[:split_point]
                validation_data = subset.iloc[split_point:]
                self.train_dataset = self.convert_dataset(train_data)
                self.validation_dataset = self.convert_dataset(validation_data)

        if stage == "test":
            if self.usecase == "target":
                test_data = pd.read_csv(self.datamodule_configs["file_paths"]["test"])
                self.test_dataset = self.convert_dataset(test_data)
            else:
                raise NotImplementedError # I should not never call this stage in the context of this project.
            
    

    def convert_dataset(self, data: pd.DataFrame) -> Dataset:
        """Converting a pandas dataframe to a pytorch tensor dataset with the smiles represented as what is specified in self.datamodule_configs["representation"]

        Args:
            data (pd.DataFrame): Dataframe with smiles and label columns

        Raises:
            NotImplementedError: Transforming smiles to representation specified in self.datamodule_configs["representation"] is not implemented

        Returns:
            Dataset: pytorch dataset with smiles represented as what is specified in self.datamodule_configs["representation"]
        """
        labels = data["label"].to_numpy()
        if self.datamodule_configs["representation"] == "ECFP_4" or self.datamodule_configs["representation"] == "ECFP_6":
            molecules = [Chem.MolFromSmiles(smiles) for smiles in data["smiles"]]
            if self.datamodule_configs["representation"] == "ECFP_4":
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            else: 
                mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
            self.input_vec_dim = 2048
            features = np.array(
                [mfpgen.GetFingerprintAsNumPy(molecule) for molecule in molecules]
            )  # (N, 2048)
            features_tensor = torch.tensor(features, dtype=torch.float)
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            dataset = TensorDataset(features_tensor, labels_tensor)
        elif self.datamodule_configs["representation"] == "MACCS":
            molecules = [Chem.MolFromSmiles(smiles) for smiles in data["smiles"]]
            maccs_keys = np.array([GetMACCSKeysFingerprint(mol) for mol in molecules])
            maccs_tensor = torch.tensor(maccs_keys, dtype=torch.float)
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            self.input_vec_dim = 167
            dataset = TensorDataset(maccs_tensor, labels_tensor)
        elif self.datamodule_configs["representation"] == "graph":
            molecules = [[Chem.MolFromSmiles(smiles)] for smiles in data["smiles"]] # chemprop wants mols as list of list with len(outer list) = number datapoints
            dataset = GraphDataset(molecules=molecules, labels=labels)
        else:
            raise NotImplementedError
        return dataset
    
    def train_dataloader(self) -> DataLoader:
        if self.datamodule_configs["representation"] == "graph":
            collate_function = mol_collate_fn
        else:
            collate_function = None
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.datamodule_configs["batch_size"],
            shuffle=True,
            num_workers=self.datamodule_configs["num_workers"],
            prefetch_factor=self.datamodule_configs["prefetch_factor"],
            collate_fn=collate_function,
        )

    def val_dataloader(self) -> DataLoader:
        if self.datamodule_configs["representation"] == "graph":
            collate_function = mol_collate_fn
        else:
            collate_function = None
        assert self.validation_dataset is not None
        return DataLoader(
            self.validation_dataset,
            batch_size=self.datamodule_configs["batch_size"],
            shuffle=False,
            num_workers=self.datamodule_configs["num_workers"],
            prefetch_factor=self.datamodule_configs["prefetch_factor"],
            collate_fn=collate_function,
        )

    def test_dataloader(self) -> DataLoader:
        if self.datamodule_configs["representation"] == "graph":
            collate_function = mol_collate_fn
        else:
            collate_function = None
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.datamodule_configs["batch_size"],
            shuffle=False,
            num_workers=self.datamodule_configs["num_workers"],
            prefetch_factor=self.datamodule_configs["prefetch_factor"],
            collate_fn=collate_function,
        )


    def population_dataloader(self) -> DataLoader:
        # create all datasets
        train_target_dataset = self.convert_dataset(pd.read_csv(self.datamodule_configs["file_paths"]["train"]))
        val_target_dataset = self.convert_dataset(pd.read_csv(self.datamodule_configs["file_paths"]["validation"])) # TODO: should include?
        test_target_dataset = self.convert_dataset(pd.read_csv(self.datamodule_configs["file_paths"]["test"]))
        aux_dataset = self.convert_dataset(pd.read_csv(self.datamodule_configs["file_paths"]["population"]))
        datasets = [train_target_dataset, val_target_dataset, test_target_dataset, aux_dataset]
        overall_dataset = ConcatDataset(datasets)
        if self.datamodule_configs["representation"] == "graph":
            collate_function = mol_collate_fn
        else:
            collate_function = None
        return DataLoader(
            overall_dataset,
            batch_size=1, # TODO: for now only one, later adapt to higher for more efficient evaluation
            shuffle=True,
            num_workers=self.datamodule_configs["num_workers"],
            prefetch_factor=self.datamodule_configs["prefetch_factor"],
            collate_fn=collate_function,           
        )