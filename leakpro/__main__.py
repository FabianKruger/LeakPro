"""Main script to run LEAKPRO on a target model."""

import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from leakpro.mia_attacks.attack_scheduler import AttackScheduler
from leakpro.reporting.utils import prepare_priavcy_risk_report
from leakpro.adapters.utils import get_config_from_moreno_result_folder

from leakpro.adapters.ames_data_module import AmesDataModule
from moreno.models.MPNN import MPNNLightning
from moreno.models.MLP import MLPLightningModel


def setup_log(name: str, save_file: bool, directory: str = '.') -> logging.Logger:
    """Generate the logger for the current run.

    Args:
    ----
        name (str): Logging file name.
        save_file (bool): Flag about whether to save to file.

    Returns:
    -------
        logging.Logger: Logger object for the current run.

    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    # Console handler for output to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    my_logger.addHandler(console_handler)

    if save_file:
        filename = f"{directory}/log_{name}.log"
        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        my_logger.addHandler(log_handler)

    return my_logger


if __name__ == "__main__":

    path_of_this_file = Path(__file__).resolve()
    parent_directory = path_of_this_file.parent.parent
    config_path = str(parent_directory / "config" / "custom_config.yml")
    with open(config_path, "rb") as f:
        configs = yaml.safe_load(f)

    # autofill configuration for my use case
    if configs["moreno_result_folder"]["folder_path"] is not None:
        configs = get_config_from_moreno_result_folder(configs)

    # Set the random seed, log_dir and inference_game
    torch.manual_seed(configs["run"]["random_seed"])
    np.random.seed(configs["run"]["random_seed"])
    random.seed(configs["run"]["random_seed"])

    log_dir = configs["run"]["log_dir"]

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_dir = f"{log_dir}/{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_log("time_analysis", configs["run"]["time_log"], directory=log_dir)
    start_time = time.time()

    # ------------------------------------------------
    # Prepare some variables

    data_module = AmesDataModule(configs)
    data_module.setup(stage="fit")
    with open(configs["LightningModule"]["hyperparameter_file_path"], "rb") as f:
        hparams = yaml.safe_load(f)
    # save for later access
    configs["LightningModule"]["hparams"] = hparams

    if configs["LightningModule"]["model_type"] == "MLP":
        model_class = MLPLightningModel
        configs["LightningModule"]["hparams"]["input_vec_dim"] = data_module.input_vec_dim
    elif configs["LightningModule"]["model_type"] == "MPNN":
        model_class = MPNNLightning
    else:
        raise ValueError(f"Model type {configs['LightningModule']['model_type']} not supported.")
    target_model = model_class.load_from_checkpoint(configs["LightningModule"]["target_model_path"])

    # ------------------------------------------------
    # LEAKPRO starts here

    attack_scheduler = AttackScheduler(
        data_module,
        model_class,
        target_model,
        configs,
        log_dir,
        logger,
    ) 
    audit_results = attack_scheduler.run_attacks()

    logger.info(str(audit_results["rmia"]["result_object"]))

    report_log = configs["audit"]["report_log"]
    privacy_game = configs["audit"]["privacy_game"]
    n_shadow_models = configs["audit"]["num_shadow_models"]

    prepare_priavcy_risk_report(
            log_dir,
            [audit_results["rmia"]["result_object"]],
            configs["audit"],
            save_path=f"{log_dir}/{report_log}/{privacy_game}/ns_{n_shadow_models}",
        )
