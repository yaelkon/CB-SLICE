
import os
import torch
import wandb
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from datasets import get_data
from models.models import create_model
from models.losses import create_loss
from utils.metrics import Custom_Metrics, Population_Metrics, Simple_Metrics
from utils.utils import reset_random_seeds
from utils.training import (
        validate_one_epoch_cbm,
        validate_one_epoch_dnn,
)


def test(config: DictConfig):
    
    gen = reset_random_seeds(config.seed)
    
    # Setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Additional info when using cuda
    if device.type == "cuda":
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")

    # Wandb
    os.environ["WANDB_CACHE_DIR"] = os.path.join(
        Path(__file__).absolute().parent, "wandb", ".cache", "wandb"
    )  # S.t. on slurm, artifacts are logged to the right place
    print("Cache dir:", os.environ["WANDB_CACHE_DIR"])
    wandb.init(
        project=config.logging.project,
        reinit=True,
        # entity=config.logging.entity,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.logging.mode,
        name=config.logging.experiment_name,
        tags=config.logging.tags,
    )
    if config.logging.mode in ["online", "disabled"]:
        wandb.run.name = wandb.run.name.split("-")[-1] + "-" + config.experiment_name
    elif config.logging.mode == "offline":
        wandb.run.name = config.experiment_name
    else:
        raise ValueError("wandb needs to be set to online, offline or disabled.")

    # ---------------------------------
    #       Prepare data and model
    # ---------------------------------
    train_loader, val_loader, test_loader = get_data(
        config,
        config.data,
        gen,
    )
    if config.stage == "val":
        data_loader = val_loader
    elif config.stage == "test":
        data_loader = test_loader
    elif config.stage == "train":
        data_loader = train_loader
    else:
        raise ValueError("stage must be either train, val or test")

    # Initialize model and training objects
    model = create_model(config)
    model = model.to(device)
    model.load_state_dict(torch.load(config.model.pretrained_model_path, weights_only=False))
    print("Model loaded from:", config.model.pretrained_model_path)
    loss_fn = create_loss(config)

    if config.model.model == "dnn":
        validate_one_epoch = validate_one_epoch_dnn
        metrics = Simple_Metrics(
            n_populations=len(config.data.subpopulations),
            device=device,
        )
    else:
        validate_one_epoch = validate_one_epoch_cbm
        metrics = Custom_Metrics(
            n_concepts=config.data.num_concepts,
            device=device,
            concept_loss=config.model.concept_loss,
        )

    population_metrics = None
    if config.data.get("subpopulations", None) is not None and config.model.model == "cbm":
        population_metrics = Population_Metrics(
            n_concepts=config.data.num_concepts,
            n_populations=len(config.data.subpopulations),
            device=device,
        ).to(device)

    validate_one_epoch(
        data_loader=data_loader,
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        epoch=0,
        config=config,
        device=device,
        population_metrics=population_metrics,
        save_eval_df=True,
    )
    

def main(config: DictConfig):
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    test(config)


if __name__ == "__main__":
    EXP_PATH = './experiments/cbm/Waterbirds/20251204-110018_CBM_debugging_code_flow/'
    DATASET_SPLIT = "val"
    with open(os.path.join(EXP_PATH, "config.yaml"), "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)
        experiment_config = DictConfig(experiment_config)
    experiment_config.logging.mode = "disabled"
    experiment_config.model.pretrained_model_path = os.path.join(EXP_PATH, "model_last.pth")
    experiment_config.stage = DATASET_SPLIT
    main(config=experiment_config)
