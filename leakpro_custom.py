"""Main script to run LEAKPRO on a target model."""

import logging
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

import leakpro.train as util
from leakpro import dataset, models
from leakpro.mia_attacks.attack_scheduler import AttackScheduler
from leakpro.reporting.utils import prepare_priavcy_risk_report

from leakpro.adapters.ames_data_module import AmesDataModule
from inference_attacks_on_molecules.models.MPNN import MPNNLightning
from inference_attacks_on_molecules.models.MLP import MLPLightningModel

def setup_log(name: str, save_file: bool) -> logging.Logger:
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
        filename = f"log_{name}.log"
        log_handler = logging.FileHandler(filename, mode="w")
        log_handler.setLevel(logging.INFO)
        log_handler.setFormatter(log_format)
        my_logger.addHandler(log_handler)

    return my_logger


if __name__ == "__main__":

    RETRAIN = True
    #args = "./config/adult.yaml"  # noqa: ERA001
    args = "./config/cifar10.yaml" # noqa: ERA001
    with open(args, "rb") as f:
        configs = yaml.safe_load(f)

    # Set the random seed, log_dir and inference_game
    torch.manual_seed(configs["run"]["random_seed"])
    np.random.seed(configs["run"]["random_seed"])
    random.seed(configs["run"]["random_seed"])

    # Setup logger
    log_dir = configs["run"]["log_dir"]
    logger = setup_log("time_analysis", configs["run"]["time_log"])

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    report_dir = f"{log_dir}/{configs['audit']['report_log']}"
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ------------------------------------------------
    # Create the population dataset

    data_module = AmesDataModule(configs)
    if configs["LightningModule"]["model_type"] == "MLP":
        model_class = MLPLightningModel
    elif configs["LightningModule"]["model_type"] == "MPNN":
        model_class = MPNNLightning
    else:
        raise ValueError(f"Model type {configs['LightningModule']['model_type']} not supported.")
    hparams = yaml.safe_load(configs["LightningModule"]["hyperparameter_file_path"])
    # save for later access
    configs["LightningModule"]["hparams"] = hparams
    target_model = model_class.load_from_checkpoint(configs["LightningModule"]["target_model_path"])

    # ------------------------------------------------
    # LEAKPRO starts here
    # Read in model, population, and metadata
    # data_dir = configs["data"]["data_dir"]
    # data_file = configs["data"]["dataset"]
    # dataset_path = f"{data_dir}/{data_file}.pkl"
    # with open(dataset_path, "rb") as file:
    #     population = joblib.load(file)


    # Get the target model + metadata
    

    # ------------------------------------------------
    # Now we have the target model, its metadata, and the train/test dataset
    # indices.
    # TODO: add lightning datamodule and lightning module
    attack_scheduler = AttackScheduler(
        data_module,
        model_class,
        target_model,
        configs,
        log_dir,
        logger,
    )  # TODO metadata includes indices for train and test data
    audit_results = attack_scheduler.run_attacks()

    logger.info(str(audit_results["rmia"]["result_object"]))

    report_log = configs["audit"]["report_log"]
    privacy_game = configs["audit"]["privacy_game"]
    n_shadow_models = configs["audit"]["num_shadow_models"]
    n_attack_data_size = configs["audit"]["f_attack_data_size"]

    prepare_priavcy_risk_report(
            log_dir,
            [audit_results["rmia"]["result_object"]],
            configs["audit"],
            save_path=f"{log_dir}/{report_log}/{privacy_game}/ns_{n_shadow_models}_fs_{n_attack_data_size}",
        )
