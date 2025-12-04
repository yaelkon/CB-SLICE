"""
Run this file to train models using a Hydra configuration, e.g.:
    python train.py +model=SCBM +data=CUB
"""

import os
from os.path import join
from pathlib import Path
import time

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from datasets import get_data
from utils.training import (
    freeze_module,
    create_optimizer,
    create_scheduler,
    train_one_epoch_cbm,
    validate_one_epoch_cbm,
    train_one_epoch_dnn,
    validate_one_epoch_dnn,
)
from utils.metrics import Custom_Metrics, Population_Metrics, Simple_Metrics
from utils.utils import reset_random_seeds
from models.losses import create_loss
from models.models import create_model


def train(config):
    """
    Run the experiments for SCBMs or baselines as defined in the config setting. This method will set up the device, the correct
    experimental paths, initialize Wandb for tracking, generate the dataset, train the model, evaluate the test set performance, and
    finally it will evaluate the intervention performance based on the policies and strategies defined in the config.
    All final results and validations will be stored in Wandb, while the most important ones will be also printed out in the terminal.
    If specified, the model can also be saved for further exploration.

    Parameters
    ----------
    configs: dict
        The config settings for training and validating as defined in configs or in the command line.
    """
    # ---------------------------------
    #       Setup
    # ---------------------------------
    if config.model.model == 'dnn':
        assert config.model.training_mode == 'joint', "Only joint training is supported for DNN.\nIf you want to train a DNN, set config.model.training_mode = 'joint' and use the config.model.j_epochs parameter."
    # Reproducibility
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
            Path(config.experiment_dir) / config.model.model / config.data.dataset / config.experiment_name / ex_name
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
    train_loader, val_loader, test_loader = get_data(
        config,
        config.data,
        gen,
    )

    # Numbers of training epochs
    if config.model.training_mode == "joint":
        t_epochs = config.model.j_epochs
    elif config.model.training_mode in ("sequential", "independent"):
        c_epochs = config.model.c_epochs
        t_epochs = config.model.t_epochs
    else:
        raise ValueError("Invalid training mode")
    # Initialize model and training objects
    model = create_model(config)
    
    if config.model.pretrained_model_path is not None:
        print(f"Loading pretrained model from {config.model.pretrained_model_path}")
        model.load_state_dict(
            torch.load(config.model.pretrained_model_path, map_location=device)
        )
        
    model.to(device)
    loss_fn = create_loss(config)

    # ---------------------------------
    #            Training
    # ---------------------------------
    if config.model.model == "dnn":
        validate_one_epoch = validate_one_epoch_dnn
        train_one_epoch = train_one_epoch_dnn
        metrics = Simple_Metrics(
        n_populations=len(config.data["subpopulations"]),
        device=device,
    )
    else:
        validate_one_epoch = validate_one_epoch_cbm
        train_one_epoch = train_one_epoch_cbm
        metrics = Custom_Metrics(config.data.num_concepts, device, concept_loss=config.model.concept_loss).to(device)

    print(
        "TRAINING "
        + str(config.model.model)
        + ": "
        + str(config.model.get("concept_learning", "") + "\n")
    )

    # For sequential & independent training: first stage is training of concept encoder
    if config.model.training_mode in ("sequential", "independent"):
        print("\nStarting concepts training!\n")
        mode = "c"

        # Freeze the target prediction part
        model.freeze_c()

        c_optimizer = create_optimizer(config.model, model)
        lr_scheduler = create_scheduler(config.model, c_optimizer)
 
        for epoch in range(c_epochs):
            # Validate the model periodically
            if epoch % config.model.validate_per_epoch == 0 and epoch > 0:
                print("\nEVALUATION ON THE VALIDATION SET:\n")
                validate_one_epoch(
                    val_loader, model, metrics, epoch, config, loss_fn, device
                )
            if epoch % config.model.save_model_per_epoch == 0 and not config.logging.debug_mode:
                torch.save(model.state_dict(), join(experiment_path, f"model_c_epoch:{epoch}.pth"))
            
            train_one_epoch(
                train_loader=train_loader,
                model=model,
                optimizer=c_optimizer,
                mode=mode,
                metrics=metrics,
                epoch=epoch,
                config=config,
                loss_fn=loss_fn,
                device=device,
            )
            lr_scheduler.step()

        # Prepare parameters for target training by unfreezing the target prediction part and freezing the concept encoder
        model.freeze_t()

    # Sequential vs. joint optimisation
    if config.model.training_mode in ("sequential", "independent"):
        print("\nStarting target training!\n")
        mode = "t"
    else:
        print("\nStarting joint training!\n")
        mode = "j"

    optimizer = create_optimizer(config.model, model)
    lr_scheduler = create_scheduler(config.model, optimizer)

    # If sequential & independent training: second stage is training of target predictor
    # If joint training: training of both concept encoder and target predictor
    for epoch in range(0, t_epochs):
        train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            mode=mode,
            metrics=metrics,
            epoch=epoch,
            config=config,
            loss_fn=loss_fn,
            device=device,
        )

        if epoch % config.model.validate_per_epoch == 0 and epoch > 0:
            print("\nEVALUATION ON THE VALIDATION SET:\n")
            validate_one_epoch(
                data_loader=val_loader,
                model=model,
                loss_fn=loss_fn,
                metrics=metrics,
                epoch=epoch,
                config=config,
                device=device,
                save_eval_df=False,
            )

        if epoch % config.model.save_model_per_epoch == 0 and not config.logging.debug_mode:
            torch.save(model.state_dict(), join(experiment_path, f"model_epoch:{epoch}.pth"))

        lr_scheduler.step()

    model.apply(freeze_module)

    print("\nEVALUATION ON THE VALIDATION SET:\n")
    validate_one_epoch(
        data_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        metrics=metrics,
        epoch=t_epochs,
        config=config,
        device=device,
        save_eval_df=not config.logging.debug_mode,
    )

    print("\nEVALUATION ON THE TEST SET:\n")
    validate_one_epoch(
        data_loader=test_loader,
        model=model,
        metrics=metrics,
        epoch=t_epochs,
        config=config,
        loss_fn=loss_fn,
        device=device,
        save_eval_df=not config.logging.debug_mode,
    )
    # Save the model
    if not config.logging.debug_mode:
        torch.save(model.state_dict(), join(experiment_path, "model_last.pth"))
        print("\nTRAINING FINISHED, MODEL SAVED!", flush=True)
    else:
        print("\nTRAINING FINISHED", flush=True)
    # Finish Wandb
    wandb.finish(quiet=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    train(config)


if __name__ == "__main__":
    main()
