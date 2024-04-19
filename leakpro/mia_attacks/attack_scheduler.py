"""Module that contains the AttackScheduler class, which is responsible for creating and executing attacks."""
import logging

import numpy as np
import torch
import lightning as L

from typing import Type
from leakpro.dataset import GeneralDataset
from leakpro.import_helper import Any, Dict, Self
from leakpro.mia_attacks.attack_factory import AttackFactory
from leakpro.mia_attacks.attack_objects import AttackObjects
from leakpro.mia_attacks.attack_utils import AttackUtils
from leakpro.mia_attacks.attacks.attack import AttackAbstract
from leakpro.adapters.custom_data_module import CustomDataModule


class AttackScheduler:
    """Class responsible for creating and executing attacks."""

    def __init__(  # noqa: D107, PLR0913
        self:Self,
        data_module: CustomDataModule,
        model_class: Type[L.LightningModule],
        target_model:L.LightningModule,
        configs:Dict[str, Any],
        logs_dirname:str,
        logger:logging.Logger
    ) -> None:
        self.attack_list = configs["audit"]["attack_list"]
        self.attacks = []

        attack_objects = AttackObjects(
            data_module=data_module, target_model_class=model_class, trained_target_model=target_model, configs=configs, logger=logger
        )
        attack_utils = AttackUtils(attack_objects)

        for attack_name in self.attack_list:
            try:
                attack = AttackFactory.create_attack(attack_name, attack_utils, configs)
                self.add_attack(attack)
            except ValueError as e:
                logger.info(e)

        self.logs_dirname = logs_dirname
        self.logger = logger

    def add_attack(self:Self, attack: AttackAbstract) -> None:
        """Add an attack to the list of attacks."""
        self.attacks.append(attack)

    def run_attacks(self:Self) -> Dict[str, Any]:
        """Run the attacks and return the results."""
        results = {}
        for attack, attack_type in zip(self.attacks, self.attack_list):
            self.logger.info(f"Preparing attack: {attack_type}")
            attack.prepare_attack()

            self.logger.info(f"Running attack: {attack_type}")

            result = attack.run_attack()
            results[attack_type] = {"attack_object": attack, "result_object": result}

            self.logger.info(f"Finished attack: {attack_type}")
        return results

    def identify_attacks(self:Self) -> None:
        """Identify relevant attacks based on adversary setting."""
        # Implementation goes here
        pass
