import yaml
import os
import torch
import wandb

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from datasets import get_data
from datasets.erroneous_dataset import create_erroneous_dataloader
from models.models import MixtureGaussiansCBM, create_model
from models.losses import ConceptErrorMixtureLoss, CBLoss
from utils.metrics import ErrorMixtureCustomMetrics
from utils.utils import reset_random_seeds
from utils.training import validate_one_gmm_epoch
    

def test(config: DictConfig):
    
    gen = reset_random_seeds(config.seed)
    
    # Setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    _, val_loader, test_loader = get_data(
        config,
        config.data,
        gen,
    )
    data_loader = val_loader
    
    # Initialize model and training objects
    pretrained_cbm = create_model(config)
    pretrained_cbm = pretrained_cbm.to(device)
    pretrained_cbm.load_state_dict(torch.load(config.model.pretrained_model_path, weights_only=False))
    print(f"Pretrained CBM loaded from {config.model.pretrained_model_path}")

    # Check if we should train only on erroneous examples
    train_on_erroneous_only = config.model.gmm_params.get("train_on_erroneous_only", False)
    if train_on_erroneous_only:
        print("Testing GMM only on erroneous examples from pretrained CBM...")
        data_loader = create_erroneous_dataloader(
            original_loader=data_loader,
            cbm_model=pretrained_cbm,
            device=device,
            num_classes=config.data.num_classes
        )
        print(f"Created filtered dataloader with {len(data_loader.dataset)} "
              f"erroneous examples")

    mixture_model = MixtureGaussiansCBM(
        pretrained_model=pretrained_cbm,
        n_clusters=config.model.gmm_params.n_clusters,
        n_concepts=config.data.num_concepts,
        n_features=config.model.gmm_params.n_features,
        n_classes=config.data.num_classes,
        use_task_heads=config.model.gmm_params.loss.use_task_loss,
        clustering_method=config.model.gmm_params.clustering_method,
        filtered_concepts=config.model.gmm_params.filtered_concepts,
    )

    mixture_model = mixture_model.to(device)
    mixture_model.load_state_dict(torch.load(
        config.model.gmm_params.pretrained_model_path, weights_only=False
    ))
    print(f"GMM pretrained model loaded from "
            f"{config.model.gmm_params.pretrained_model_path}")
    loss_gmm = ConceptErrorMixtureLoss(
        config=config.model.gmm_params.loss,
        concept_loss=config.model.concept_loss,
        warmup=0,
    ).to(device)
    
    loss_cbm = CBLoss(
        num_classes=config.data.num_classes,
        alpha=config.model.alpha,
        config=config.model,
    ).to(device)
    
    metrics = ErrorMixtureCustomMetrics(
        concept_loss=config.model.concept_loss,
        device=device,
    ).to(device)

    population_metrics = None
    # if config.data.get("subpopulations", None) is not None:
    #     population_metrics = Population_Metrics(
    #         n_concepts=config.data.num_concepts,
    #         n_populations=len(config.data.subpopulations),
    #         device=device,
    #     ).to(device)

    validate_one_gmm_epoch(
        loader=data_loader,
        model=mixture_model,
        metrics=metrics,
        config=config,
        loss_gmm=loss_gmm,
        loss_cbm=loss_cbm,
        device=device,
        epoch=0,
        test=False,
        population_metrics=population_metrics,
        save_mixture_slicer_df=True,
    )
    
def main(config: DictConfig):
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    test(config)


if __name__ == "__main__":
    EXP_PATH = './experiments/cbm/Waterbirds/20251204-110018_CBM_debugging_code_flow/GMM/20251204-113817_GMM_debugging_code_flow/'

    with open(os.path.join(EXP_PATH, "config.yaml"), "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)
        experiment_config = DictConfig(experiment_config)
    experiment_config.logging.mode = "disabled"
    experiment_config.model.gmm_params.pretrained_model_path = os.path.join(EXP_PATH, "model_last.pth")
    main(config=experiment_config)
