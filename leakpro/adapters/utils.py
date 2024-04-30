from pathlib import Path

def get_config_from_moreno_result_folder(configs: dict) -> dict:

    result_folder_path = Path(configs["moreno_result_folder"]["folder_path"])
    representation = configs["LightningDataModule"]["representation"]
    dataset = configs["LightningDataModule"]["dataset_name"]
    # get dataset paths
    configs["LightningDataModule"]["file_paths"]["train"] = str(result_folder_path / f"data_dir/{dataset}_random_train.csv")
    configs["LightningDataModule"]["file_paths"]["validation"] = str(result_folder_path / f"data_dir/{dataset}_random_validation.csv")
    configs["LightningDataModule"]["file_paths"]["test"] = str(result_folder_path / f"data_dir/{dataset}_random_test.csv")
    configs["LightningDataModule"]["file_paths"]["population"] = str(result_folder_path / f"data_dir/{dataset}_random_auxiliary.csv")
    # get model path
    configs["LightningModule"]["target_model_path"] = str(result_folder_path / f"results_{dataset}_{representation}/model_{representation}.ckpt")
    # get hyperparameter file path
    configs["LightningModule"]["hyperparameter_file_path"] = str(result_folder_path / f"results_{dataset}_{representation}/optimized_hyperparameters.yaml")
    # set log dir (where to save shadow models etc.)
    configs["run"]["log_dir"] = str(result_folder_path / "leakpro" / f"{dataset}_{representation}")
    # set audit dir
    configs["audit"]["report_log"] = "audit_report"
    return configs