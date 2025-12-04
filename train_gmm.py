import hydra
import os
import torch
import time
import wandb

from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from datasets import get_data
from datasets.erroneous_dataset import create_erroneous_dataloader
from models.models import create_model, MixtureGaussiansCBM
from models.losses import ConceptErrorMixtureLoss, CBLoss
from utils.training import (
    create_optimizer,
    create_scheduler,
    train_one_gmm_epoch,
    validate_one_gmm_epoch,
)
from utils.metrics import ErrorMixtureCustomMetrics
from utils.utils import reset_random_seeds


def train_gmm(config: DictConfig):
    
    gen = reset_random_seeds(config.seed)
    
    # Setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Additional info when using cuda
    if device.type == "cuda":
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")
        # Set paths
    if not config.logging.debug_mode:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        ex_name = "{}_{}".format(str(timestr), config.logging.experiment_name)
        experiment_path = (
            Path(config.experiment_dir) / "GMM" / ex_name
        )
        experiment_path.mkdir(parents=True)
        config.experiment_dir = str(experiment_path)
        print("Experiment path: ", experiment_path)

        # Save config
        with open(os.path.join(experiment_path, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)
            print(f"Config saved to {os.path.join(experiment_path, 'config.yaml')}")

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
    
    # Initialize model and training objects
    pretrained_cbm = create_model(config)
    pretrained_cbm = pretrained_cbm.to(device)
    pretrained_cbm.load_state_dict(torch.load(
        config.model.pretrained_model_path, weights_only=False
    ))
    print(f"Pretrained CBM loaded from {config.model.pretrained_model_path}")

    # Check if we should train only on erroneous examples
    train_on_erroneous_only = config.model.gmm_params.get("train_on_erroneous_only", False)
    if train_on_erroneous_only:
        print("Training GMM only on erroneous examples from pretrained CBM...")
        # Create filtered dataloader with only erroneous examples
        val_loader = create_erroneous_dataloader(
            original_loader=val_loader,
            cbm_model=pretrained_cbm,
            device=device,
            num_classes=config.data.num_classes
        )
        print(f"Created filtered dataloader with {len(val_loader.dataset)} "
              f"erroneous examples")

    #########################################################
    #                   Build GMM Model                      #
    #########################################################
    mixture_model = MixtureGaussiansCBM(
        pretrained_model=pretrained_cbm,
        n_clusters=config.model.gmm_params.n_clusters,
        n_concepts=config.data.num_concepts,
        n_features=config.model.gmm_params.n_features,
        n_classes=config.data.num_classes,
        use_task_heads=config.model.gmm_params.loss.use_task_loss,
        clustering_method=config.model.gmm_params.clustering_method,
        filtered_concepts=config.model.gmm_params.filtered_concepts,
        # config.model.gaussian_dim if isinstance(config.model.get("gaussian_dim", None), int) else config.data.num_concepts,
    )
    mixture_model = mixture_model.to(device)
    if config.model.gmm_params.get("pretrained_model_path") is not None:
        mixture_model.load_state_dict(torch.load(
            config.model.gmm_params.pretrained_model_path, weights_only=False
        ))
        print(f"GMM pretrained model loaded from "
              f"{config.model.gmm_params.pretrained_model_path}")

    # Build GMM loss
    loss_gmm = ConceptErrorMixtureLoss(
        config=config.model.gmm_params.loss,
        concept_loss=config.model.concept_loss,
    ).to(device)
    # Build CBM loss for evaluation purposes
    loss_cbm = CBLoss(
        num_classes=config.data.num_classes,
        alpha=config.model.alpha,
        config=config.model,
    ).to(device)
   
    # Create optimizer and scheduler
    optimizer = create_optimizer(config.model, mixture_model)
    scheduler = create_scheduler(config.model, optimizer)

    metrics = ErrorMixtureCustomMetrics(
        concept_loss=config.model.concept_loss,
        device=device,
    ).to(device)

    # ---------------------------------
    #            Training
    # ---------------------------------
    print(
        f"TRAINING GMM FOR: {config.model.model}\n"
        + f"Num of Gaussians: {config.model.gmm_params.n_clusters}\n"
    )

    for epoch in range(0, config.model.gmm_params.g_epochs):
        train_one_gmm_epoch(
            epoch=epoch,
            model=mixture_model,
            train_loader=val_loader,
            optimizer=optimizer,
            loss_func=loss_gmm,
            config=config,
            device=device,
            metrics=metrics,
        )
        scheduler.step()

        if epoch % config.model.validate_per_epoch == 0 and epoch > 0:
            print("\nEVALUATION ON THE VALIDATION SET:\n")
            validate_one_gmm_epoch(
                loader=test_loader,
                model=mixture_model,
                metrics=metrics,
                config=config,
                loss_gmm=loss_gmm,
                loss_cbm=loss_cbm,
                device=device,
                epoch=epoch,
                test=True,
                population_metrics=None,
                save_mixture_slicer_df=False,
            )

        if (not config.logging.debug_mode and
                (epoch % config.model.save_model_per_epoch == 0) and
                (epoch > 0)):
            print(f"Saving model at epoch {epoch}")
            torch.save(
                mixture_model.state_dict(),
                os.path.join(config.experiment_dir, f"model_epoch:{epoch}.pth"),
            )

    if not config.logging.debug_mode:
        print("Saving model_last")
        torch.save(
            mixture_model.state_dict(),
            os.path.join(config.experiment_dir, f"model_last.pth"),
        )

    # ---------------------------------
    print("EVALUATION ON THE VALIDATION SET:")
    validate_one_gmm_epoch(
        loader=val_loader,
        model=mixture_model,
        metrics=metrics,
        config=config,
        loss_gmm=loss_gmm,
        loss_cbm=loss_cbm,
        device=device,
        epoch=epoch,
        test=False,
        population_metrics=None,
        save_mixture_slicer_df=not config.logging.debug_mode,
    )

    print("EVALUATION ON THE TEST SET:")
    validate_one_gmm_epoch(
        loader=test_loader,
        model=mixture_model,
        metrics=metrics,
        config=config,
        loss_gmm=loss_gmm,
        loss_cbm=loss_cbm,
        device=device,
        epoch=epoch,
        test=True,
        population_metrics=None,
        save_mixture_slicer_df=not config.logging.debug_mode,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    train_gmm(config)


if __name__ == "__main__":
    main()
