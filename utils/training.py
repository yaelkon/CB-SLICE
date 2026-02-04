"""
Utility functions for training.
"""
import torch
import os
import wandb
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from utils.plotting import SubpoopulationPlotter
from utils.utils import torch_to_numpy, tensor_to_serializable
from baselines.utils.metrics import (
    precision_at_k,
    compute_slice_prioritisation_scores,
    find_slice_key_concepts,
)
from baselines.utils.plotting import plot_slices
from baselines.utils.utils import data_preprocessing


def train_one_epoch_cbm(
    train_loader,
    model,
    loss_fn,
    optimizer,
    mode,
    metrics,
    epoch,
    config,
    device,
):
    """
    Train a baseline method for one epoch.

    This function trains the CEM/AR/CBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    if config.model.training_mode in ("sequential", "independent"):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):  
        if config["model"]["limit_train_batches"] is not None and k >= config["model"]["limit_train_batches"]:
            break

        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        if config.model.training_mode == "independent" and mode == "t":
            output = model(batch_features, epoch, concepts_true)
        else:
            output = model(batch_features, epoch)

        concepts_pred_probs, target_pred_logits = output["c_prob"], output["y_pred_logits"]
        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()
        # Compute the loss
        loss_dict = loss_fn(
            concepts_pred_probs=concepts_pred_probs,
            concepts_true=concepts_true,
            target_pred_logits=target_pred_logits,
            target_true=target_true,
            concept_pred_logits=output.get("c_logits", None),
        )

        target_loss, concepts_loss, total_loss = loss_dict["target_loss"], loss_dict["concepts_loss"], loss_dict["total_loss"]
        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            concepts_loss.backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits,
            concepts_true,
            concepts_pred_probs,
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    total_loss = metrics_dict["total_loss"]
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()

    return total_loss

def validate_one_epoch_cbm(
    data_loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    population_metrics=None,
    save_eval_df=False,
):
    """
    Validate a baseline method for one epoch.

    This function evaluates the CEM/AR/CBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
    """
    stage = data_loader.dataset.stage
    model.eval()

    if save_eval_df:
        # Create a DataFrame to store the concepts predictions in the structure of: 'embeddings', 'target', 'pred_probs'
        eval_list = []
    
    with torch.no_grad():
        for k, batch in enumerate(
            tqdm(data_loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            if config["model"]["limit_val_batches"] is not None and k >= config["model"]["limit_val_batches"]:
                break
            
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)

            output = model(
                batch_features, validation=True
            )
            concepts_pred_probs, y_pred_logits = output["c_prob"], output["y_pred_logits"]
 
            losses_dict = loss_fn(
                concepts_pred_probs=concepts_pred_probs,
                concepts_true=concepts_true,
                target_pred_logits=y_pred_logits,
                target_true=target_true,
                concept_pred_logits=output.get("c_logits", None),
            )
            target_loss, concepts_loss, total_loss = losses_dict["target_loss"], losses_dict["concepts_loss"], losses_dict["total_loss"]

            # Store predictions
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                y_pred_logits,
                concepts_true,
                concepts_pred_probs,
            )

            if population_metrics is not None:
                subpopulation_indices = batch["extra_dict"]["population_idx"].to(device)
                population_metrics.update(
                    target_loss=losses_dict["per_sample_target_loss"],
                    concepts_loss=losses_dict["per_concept_loss"],
                    y_true=target_true,
                    y_pred_logits=y_pred_logits,
                    c_true=concepts_true,
                    c_pred_probs=concepts_pred_probs,
                    subpopulation_indices=subpopulation_indices,
                )

            if save_eval_df:
                y_preds, y_scores = calc_preds_and_scores(y_pred_logits)
                for i in range(target_true.shape[0]):
                    eval_dict = {
                        "img_id": torch_to_numpy(batch["img_code"][i]),
                        "labels": torch_to_numpy(target_true[i]),
                        "concepts": torch_to_numpy(concepts_true[i]),
                        "population_idx": torch_to_numpy(batch["extra_dict"]["population_idx"][i]),
                        "y_preds": torch_to_numpy(y_preds[i]),
                        "y_scores": torch_to_numpy(y_scores[i]),
                        "y_logits": torch_to_numpy(y_pred_logits[i]),
                        "c_scores": torch_to_numpy(concepts_pred_probs[i]),
                        "embeddings": torch_to_numpy(output["c_logits"][i]),
                        "target_loss": torch_to_numpy(losses_dict["per_sample_target_loss"][i]),
                        "concepts_loss": torch_to_numpy(losses_dict["per_concept_loss"][i]),
                    }
                    if "attribute" in batch["extra_dict"]:
                        eval_dict["attribute"] = batch["extra_dict"]["attributes"][i]
                    if "population_name" in batch["extra_dict"]:
                        eval_dict["population_name"] = batch["extra_dict"]["population_name"][i]
                    if "origin_label" in batch["extra_dict"]:
                        eval_dict["origin_label"] = torch_to_numpy(batch["extra_dict"]["origin_label"][i])
                    if "img_path" in batch["extra_dict"]:
                        eval_dict["img_path"] = batch["extra_dict"]["img_path"][i]
                    if "img" in batch["extra_dict"]:
                        eval_dict["img"] = batch["extra_dict"]["img"][i]
                    eval_list.append(eval_dict)

    metrics_dict = metrics.compute(validation=True, config=config)

    if save_eval_df:
        results_save_path = os.path.join(config.experiment_dir, "Evaluations")
        os.makedirs(results_save_path, exist_ok=True)
        # Save the mixture slicer DataFrame to a pkl file
        df = pd.DataFrame(eval_list)
        df_name = stage + "_eval_df.pkl"
        df.to_pickle(os.path.join(results_save_path, df_name))
    
        if population_metrics is not None:
            # Compute and plot subpopulation statistics
            population_metrics_dict = population_metrics.compute(validation=True)
            metrics_dict["worst_group_accuracy"] = round(min(population_metrics_dict["task_accuracy"]), 4)
            
            subpop_plotter = SubpoopulationPlotter(
                population_metrics=population_metrics_dict,
                subpopulations_str2idx=data_loader.dataset.subpopulations_dict,
                log_scale=False,
                stage=stage,
                save_path=results_save_path if not config.logging.debug_mode else "",
            )

            subpop_plotter.plot(plot_uncertainty=False)

            population_metrics.reset()

        # Save metrics to a json file
        metrics_dict_serializable = tensor_to_serializable(metrics_dict)
        with open(os.path.join(results_save_path, f"{stage}_metrics.json"), "w") as f:
            json.dump(metrics_dict_serializable, f)

    if stage in ["val", "train"]:
        wandb.log({f"{stage}/{k}": v for k, v in metrics_dict.items()})
        prints = f"Epoch {epoch}, Validation: "
    elif stage == "test":
        wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
        prints = f"Test: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()

    metrics.reset()
    return


def train_one_epoch_dnn(
        train_loader,
        model,
        optimizer,
        metrics,
        epoch,
        config,
        loss_fn,
        device,
        mode=None,
):
    """
    Train the DNN for one epoch.
    """

    model.train()
    metrics.reset()
    
    for k, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
        
        if config["model"]["limit_train_batches"] is not None and k >= config["model"]["limit_train_batches"]:
            break
        
        x, y = batch["features"].to(device), batch["labels"].to(device)
        subpopulation_indices = batch["extra_dict"].get("population_idx", None).to(device)
        
        output = model(x)
        optimizer.zero_grad()
        loss_dict = loss_fn(
                y_true=y,
                y_pred_logits=output["logits"],
        )
        loss_dict["loss"].backward()
        optimizer.step()

        metrics.update(
                loss=loss_dict["per_sample_target_loss"].detach(), 
                y_true=y,
                y_pred_logits=output["logits"].detach(),
                subpopulation_indices=subpopulation_indices,
        )

    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()


def validate_one_epoch_dnn(
        data_loader,
        model,
        loss_fn,
        metrics,
        epoch,
        config,
        device,
        save_eval_df=False,
        population_metrics=None,
):
    """
    Validate the DNN on the validation set.
    """
    stage = data_loader.dataset.stage
    if save_eval_df:
        eval_list = []

    model.eval()
    metrics.reset()
    with torch.no_grad():       
        for k, batch in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            
            if config.model.get("limit_val_batches", None) is not None and k >= config.model.limit_val_batches:
                break
                
            x, y = batch["features"].to(device), batch["labels"].to(device)
            subpopulation_indices = batch["extra_dict"].get("population_idx", None).to(device)
            output = model(x)
            loss_dict = loss_fn(
                y_true=y,
                y_pred_logits=output["logits"],
            )
            metrics.update(
                loss=loss_dict["per_sample_target_loss"].detach(),
                y_true=y,
                y_pred_logits=output["logits"].detach(),
                subpopulation_indices=subpopulation_indices,
            )

            if save_eval_df:
                y_preds, y_scores = calc_preds_and_scores(output["logits"])
                for i in range(y.shape[0]):
                    eval_dict = {
                        "img_id": torch_to_numpy(batch["img_code"][i]),
                        "labels": torch_to_numpy(y[i]),
                        "population_idx": torch_to_numpy(subpopulation_indices[i]),
                        "y_preds": torch_to_numpy(y_preds[i]),
                        "y_scores": torch_to_numpy(y_scores[i]),
                        "embeddings": torch_to_numpy(output["embeddings"][i]),
                        "loss": torch_to_numpy(loss_dict["per_sample_target_loss"][i]),
                    }
                    
                    if "population_name" in batch["extra_dict"]:
                        eval_dict["population_name"] = batch["extra_dict"]["population_name"][i]
                    if "img_path" in batch["extra_dict"]:
                        eval_dict["img_path"] = batch["extra_dict"]["img_path"][i]
                    if "img" in batch["extra_dict"]:
                        eval_dict["img"] = batch["extra_dict"]["img"][i]
                    
                    eval_list.append(eval_dict)

    metrics_dict = metrics.compute()

    if save_eval_df:
        results_save_path = os.path.join(config.experiment_dir, "Evaluations")
        os.makedirs(results_save_path, exist_ok=True)
        df = pd.DataFrame(eval_list)
        df.to_pickle(os.path.join(results_save_path, f"{stage}_eval_df.pkl"))
        print(f"Evaluation results saved to {results_save_path}")
        
        if config.data.get("subpopulations", None) is not None:
            sub_pop_metric_dict = dict(metrics_dict)
            n_populations = len(data_loader.dataset.subpopulations_dict)
            sub_pop_metric_dict["task_accuracy"] = [sub_pop_metric_dict[f"task_accuracy_per_population_{p}"] for p in range(n_populations)]
            subpop_plotter = SubpoopulationPlotter(
                population_metrics=sub_pop_metric_dict,
                subpopulations_str2idx=data_loader.dataset.subpopulations_dict,
                log_scale=False,
                stage=stage,
                save_path=results_save_path if not config.logging.debug_mode else "",
            )
            subpop_plotter.plot_subpopulation_accuracies(concept_flag=False)

    wandb.log({f"{stage}/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, {stage.capitalize()}: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    
    metrics.reset()

def train_one_gmm_epoch(
    epoch,
    model,
    train_loader,
    optimizer,
    loss_func,
    config,
    device,
    metrics,
):
    """
    Train the model for one epoch.
    """
    model.train()
    metrics.reset()

    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0)):

        if config["model"]["limit_train_batches"] is not None and i >= config["model"]["limit_train_batches"]:
            break

        x, labels = batch["features"].to(device), batch["labels"].to(device)
        c_true = batch["concepts"].to(device)
        if model.filtered_concepts is not None:
            c_true = c_true[:, model.filtered_concepts]
        population_idx = batch["extra_dict"]["population_idx"].to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x, labels=labels, cbm_validation=True)

        if config.model.concept_loss == 'bce':
            c_preds = (outputs["cbm_outputs"]["c_prob"] > 0.5).float()
        elif config.model.concept_loss == 'ce':
            c_preds = outputs["cbm_outputs"]["c_prob"].argmax(dim=-1).float()
            # Convert to one-hot encoding
            c_preds = torch.nn.functional.one_hot(
                c_preds.long(), num_classes=config.data.num_concepts
            ).float()

        loss_dict = loss_func(
            pred_log_likelihood_matrix=outputs["log_lokelihood_matrix"],
            concepts_gt=c_true,
            concepts_preds=c_preds,
            concepts_logits=outputs["c_true_logits"],
            concepts_pred_logits=outputs["c_pred_logits"],
            y_true=labels,
            y_logits=outputs["cbm_outputs"]["y_pred_logits"],
            y_true_logits=outputs["y_pred_logits_1"],
            y_pred_logits=outputs["y_pred_logits_2"],
        )

        loss = loss_dict["total_loss"]
        loss.backward()

        # Clip gradients to avoid exploding gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update metrics
        cluster_idx = torch.argmax(outputs["assignment_probs"], dim=-1)
        metrics.update(
            concept_pred_loss=loss_dict["concept_pred_loss"].detach(),
            concepts_loss=loss_dict["concept_gt_loss"].detach(),
            target_gt_loss=loss_dict["target_gt_loss"].detach(),
            target_pred_loss=loss_dict["target_pred_loss"].detach(),
            mixture_loss=loss_dict["mixture_loss"].detach(),
            total_loss=loss_dict["total_loss"].detach(),
            c_true=c_true,
            c_true_star_logits=outputs["c_true_logits"].detach(),
            c_preds=c_preds,
            c_pred_star_logits=outputs["c_pred_logits"].detach(),
            y_true=labels,
            y_logits=outputs["cbm_outputs"]["y_pred_logits"],
            y_true_star_logits=outputs["y_pred_logits_1"],
            y_pred_star_logits=outputs["y_pred_logits_2"],
            cluster_idx=cluster_idx,
            population_idx=population_idx,
        )

    # Calculate and log metrics
    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()


def validate_one_gmm_epoch(
    loader,
    model,
    metrics,
    config,
    loss_gmm,
    loss_cbm,
    device,
    epoch=0,
    test=False,
    population_metrics=None,
    save_mixture_slicer_df=False,
):
    
    model.eval()

    if save_mixture_slicer_df:
        # Create a DataFrame to store the concepts predictions in the structure of: 'embeddings', 'target', 'pred_probs'
        mixture_slicer_list = []
    
    with torch.no_grad():
        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            x, labels = batch["features"].to(device), batch["labels"].to(device)
            c_true = batch["concepts"].to(device)
            if model.filtered_concepts is not None:
                c_true = c_true[:, model.filtered_concepts]
            population_idx = batch["extra_dict"]["population_idx"].to(device)
            
            # Forward pass
            outputs = model(x, cbm_validation=True)

            if config.model.concept_loss == 'bce':
                c_preds = (outputs["cbm_outputs"]["c_prob"] > 0.5).float()
            elif config.model.concept_loss == 'ce':
                c_preds = outputs["cbm_outputs"]["c_prob"].argmax(dim=-1).float()
                # Convert to one-hot encoding
                c_preds = torch.nn.functional.one_hot(c_preds.long(), num_classes=config.data.num_concepts).float()
            
            loss_gmm_dict = loss_gmm(
                pred_log_likelihood_matrix=outputs["log_lokelihood_matrix"],
                concepts_gt=c_true,
                concepts_preds=c_preds,
                concepts_logits=outputs["c_true_logits"],
                concepts_pred_logits=outputs["c_pred_logits"],
                y_true=labels,
                y_logits=outputs["cbm_outputs"]["y_pred_logits"],
                y_true_logits=outputs["y_pred_logits_1"],
                y_pred_logits=outputs["y_pred_logits_2"],
            )

            loss_cbm_dict = loss_cbm(
                concepts_pred_probs=outputs["cbm_outputs"]["c_prob"],
                concepts_true=c_true,
                target_pred_logits=outputs["cbm_outputs"]["y_pred_logits"],
                target_true=labels,
                concept_pred_logits=outputs["cbm_outputs"]["c_logits"],
            )

            # Update metrics
            # A problematic code fragment. I fixed it inside the update() function
            # c_true_probs = torch.sigmoid(outputs["c_true_logits"].detach())
            # c_pred_probs = torch.sigmoid(outputs["c_pred_logits"].detach())
            cluster_ids = torch.argmax(outputs["assignment_probs"], dim=-1)
            metrics.update(
                concept_pred_loss=loss_gmm_dict["concept_pred_loss"].detach(),
                concepts_loss=loss_gmm_dict["concept_gt_loss"].detach(),
                target_gt_loss=loss_gmm_dict["target_gt_loss"].detach(),
                target_pred_loss=loss_gmm_dict["target_pred_loss"].detach(),
                mixture_loss=loss_gmm_dict["mixture_loss"].detach(),
                total_loss=loss_gmm_dict["total_loss"].detach(),
                c_true=c_true,
                c_true_star_logits=outputs["c_true_logits"].detach(),
                c_preds=c_preds.detach(),
                c_pred_star_logits=outputs["c_pred_logits"].detach(),
                y_true=labels,
                y_logits=outputs["cbm_outputs"]["y_pred_logits"],
                y_true_star_logits=outputs["y_pred_logits_1"],
                y_pred_star_logits=outputs["y_pred_logits_2"],
                population_idx=population_idx,
                cluster_idx=cluster_ids,
            )

            if save_mixture_slicer_df:
                y_pred_logits = outputs["cbm_outputs"]["y_pred_logits"]
                y_pred, y_pred_probs = calc_preds_and_scores(y_pred_logits)

                if outputs["y_pred_logits_1"] is not None and outputs["y_pred_logits_2"] is not None:
                    y_true_star, _ = calc_preds_and_scores(outputs["y_pred_logits_1"])
                    y_pred_star, _ = calc_preds_and_scores(outputs["y_pred_logits_2"])

                if config.model.concept_loss == 'bce':
                    c_true_probs = torch.sigmoid(outputs["c_true_logits"].detach())
                    c_pred_probs = torch.sigmoid(outputs["c_pred_logits"].detach())
                elif config.model.concept_loss == 'ce':
                    c_true_probs = torch.softmax(outputs["c_true_logits"].detach(), dim=-1)
                    c_pred_probs = torch.softmax(outputs["c_pred_logits"].detach(), dim=-1)
                else:
                    raise ValueError("Unknown concept loss type")
                  
                # Calculate the ECTP for each sample
                ectp = model.model.calculate_ectp(x)
                ecca = model.calculate_ecca(x)
                if model.filtered_concepts is not None:
                    ectp = ectp[:, model.filtered_concepts]
                
                for i in range(labels.shape[0]):
                    mixture_slicer_dict = {
                        "img_id": torch_to_numpy(batch["img_code"][i]),
                        "labels": torch_to_numpy(labels[i]),
                        "concepts": torch_to_numpy(c_true[i]),
                        "population_idx": torch_to_numpy(batch["extra_dict"]["population_idx"][i]),
                        "y_preds": torch_to_numpy(y_pred[i]),
                        "y_scores": torch_to_numpy(y_pred_probs[i]),
                        "c_scores": torch_to_numpy(outputs["cbm_outputs"]["c_prob"][i]),
                        "ectp_scores": torch_to_numpy(ectp[i]),
                        "embeddings": torch_to_numpy(outputs["cbm_outputs"]["c_logits"][i]),
                        "target_loss": torch_to_numpy(loss_cbm_dict["per_sample_target_loss"][i]),
                        "concepts_loss": torch_to_numpy(loss_cbm_dict["per_concept_loss"][i]),
                        "GMM_stats:cluster_id": torch_to_numpy(cluster_ids[i]),
                        "GMM_stats:cluster_probs": np.round(torch_to_numpy(outputs["assignment_probs"][i]), 3),
                        "GMM_stats:c_true_star_probs": torch_to_numpy(c_true_probs[i]),
                        "GMM_stats:c_pred_star_probs": torch_to_numpy(c_pred_probs[i]),
                        "GMM_stats:ecca_score": torch_to_numpy(ecca[i]),
                    }
                    if "attribute" in batch["extra_dict"]:
                        mixture_slicer_dict["attribute"] = batch["extra_dict"]["attributes"][i]
                    if "population_name" in batch["extra_dict"]:
                        mixture_slicer_dict["population_name"] = batch["extra_dict"]["population_name"][i]
                    if "origin_label" in batch["extra_dict"]:
                        mixture_slicer_dict["origin_label"] = torch_to_numpy(batch["extra_dict"]["origin_label"][i])
                    if outputs["y_pred_logits_1"] is not None:
                        mixture_slicer_dict["GMM_stats:y_true_star"] = torch_to_numpy(y_true_star[i])
                        mixture_slicer_dict["GMM_stats:y_pred_star"] = torch_to_numpy(y_pred_star[i])
                    if "img_path" in batch["extra_dict"]:
                        mixture_slicer_dict["img_path"] = batch["extra_dict"]["img_path"][i]
                    if "img" in batch["extra_dict"]:
                        mixture_slicer_dict["img"] = batch["extra_dict"]["img"][i]
                    mixture_slicer_list.append(mixture_slicer_dict)

    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)

    if save_mixture_slicer_df:
        # Save the mixture slicer DataFrame to a pickle file in the intended directory
        save_path = os.path.join(*[config.experiment_dir, "MixtureSlicer", loader.dataset.stage])
        os.makedirs(save_path, exist_ok=True)

        df = pd.DataFrame(mixture_slicer_list)
        df_name = loader.dataset.stage + "_gmm_eval_df.pkl"
        save_file = os.path.abspath(os.path.join(save_path, df_name))
        df.to_pickle(save_file)

        # Double-check file was written and show location
        if os.path.isfile(save_file):
            print(f"Mixture slicer DataFrame saved to {save_file}")
        else:
            print(f"ERROR: Mixture slicer DataFrame was not saved! Checked path: {save_file}")

        print("Saving Slice Statistics and Plots...")
        stats_save_path = os.path.join(save_path, "results")
        os.makedirs(stats_save_path, exist_ok=True)

        preprocessed_df = data_preprocessing(df)
        # Save Slice Statistics and Plots
        precision_at_k_results = precision_at_k(
            population_ids=preprocessed_df["population_idx"],
            slices_preds=preprocessed_df["GMM_stats:cluster_id"],
            slices_probs=preprocessed_df["GMM_stats:cluster_probs"],
            k=10,
            save_dir=stats_save_path,
        )

        slice_prioritisation_scores_df = compute_slice_prioritisation_scores(
            slice_ids=preprocessed_df["GMM_stats:cluster_id"],
            slice_probs=preprocessed_df["GMM_stats:cluster_probs"],
            y_preds=preprocessed_df["y_preds"],
            embeddings=preprocessed_df["embeddings"],
            n_classes=config.data.num_classes,
            saving_dir=stats_save_path,
        )

        if hasattr(loader.dataset, "original_dataset"):
            concept_semantics = loader.dataset.original_dataset.concept_semantics
        else:
            concept_semantics = loader.dataset.concept_semantics

        find_slice_key_concepts(
            gmm_eval_dict=preprocessed_df,
            experiment_config=config,
            max_rep_concepts=10,
            semantic_concepts=concept_semantics,
            save_dir=stats_save_path,
        )

        sorted_embeddings_args = np.argsort(preprocessed_df["embeddings"][:, 0], axis=0)
        sorted_embeddings = preprocessed_df["embeddings"][sorted_embeddings_args]
        sorted_slices = preprocessed_df["GMM_stats:cluster_id"][sorted_embeddings_args]
        sorted_population_idx = preprocessed_df["population_idx"][sorted_embeddings_args]
        
        plot_slices(
            embeddings=sorted_embeddings,
            slices=sorted_slices,
            saving_path=stats_save_path,
            fig_name="discovered_error_slices_tsne_plot",
            title="Discovered Error Slices Visualisation",
            perplexity=30,
            learning_rate=100,
        )
        plot_slices(
            embeddings=sorted_embeddings,
            slices=sorted_population_idx,
            saving_path=stats_save_path,
            fig_name="gt_error_slices_tsne_plot",
            title="GT Error Slices Visualisation",
            perplexity=30,
            learning_rate=100,
        )

    if not test:
        wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
        prints = f"Epoch {epoch}, Validation: "
    else:
        wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
        prints = f"Test: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
    if population_metrics is not None:
        population_metrics.reset()

def create_optimizer(config, model):
    """
    Parse the configuration file and return a optimizer object to update the model parameters.
    """
    assert config.optimizer in [
        "sgd",
        "adam",
    ], "Only SGD and Adam optimizers are available!"

    optim_params = [
        {
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "momentum": config.momentum,
        }
    ]

    if config.optimizer == "sgd":
        return torch.optim.SGD(optim_params)
    elif config.optimizer == "adam":
        return torch.optim.Adam(optim_params)


def create_scheduler(model_config, optimizer):
    """
    Parse the configuration file and return a scheduler object to update the learning rate of the optimizer.
    """
    if model_config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=model_config.decrease_every,
            gamma=1 / model_config.lr_divisor,
        )
    else:
        raise ValueError(f"Scheduler {model_config.scheduler} not supported!")


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True


def calc_preds_and_scores(logits, threshold=0.5):
    """
    Calculate predictions and scores from logits.
    
    Args:
        logits (torch.Tensor): Logits from the model.
        threshold (float): Threshold for binary classification.
        
    Returns:
        tuple: Predictions and scores.
    """
    if logits.size(1) == 1:  # Binary classification
        scores = torch.sigmoid(logits.squeeze())
        preds = (scores > threshold).int()
    else:  # Multi-class classification
        scores = torch.softmax(logits, dim=-1)
        preds = scores.argmax(dim=-1)
    
    return preds, scores
