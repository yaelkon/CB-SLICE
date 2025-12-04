"""
Utility methods for constructing loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict


def create_loss(config):
    """
    Create and return a loss function based on the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        nn.Module: The loss function.
    """
    if config.model.model == "cbm":
        return CBLoss(
            num_classes=config.data.num_classes,
            alpha=config.model.alpha,
            config=config.model,
        )

    elif config.model.model == "dnn":
        return DNNLoss(
            num_classes=config.data.num_classes,
        )
    else:
        raise NotImplementedError


class DNNLoss(nn.Module):
    """
    Loss function for the Deep Neural Network (DNN).
    """
    def __init__(self, num_classes: int):
        super(DNNLoss, self).__init__()
        self.num_classes = num_classes

    def forward(
        self,
        y_true,
        y_pred_logits,
    ) -> Dict[str, Tensor]:
        """
        Compute the loss.

        Args:
            y_true (Tensor): Ground-truth target values.
            y_pred_logits (Tensor): Predicted target logits.

        Returns:
            Dict[str, Tensor]: Loss dictionary.
        """
        if self.num_classes == 2:
            # Logits to probs
            target_pred_probs = nn.Sigmoid()(y_pred_logits.squeeze(1))
            per_sample_target_loss = F.binary_cross_entropy(
                target_pred_probs, y_true.float(), reduction="none"
            )
        else:
            per_sample_target_loss = F.cross_entropy(
                y_pred_logits, y_true.long(), reduction="none"
            )
        loss = per_sample_target_loss.mean()

        loss_dict = {
            "loss": loss,
            "per_sample_target_loss": per_sample_target_loss,
        }
        return loss_dict
        

class CBLoss(nn.Module):
    """
    Loss function for the Concept Bottleneck Model (CBM).
    """

    def __init__(
        self,
        num_classes: int,
        # reduction: str = "mean",
        alpha: float = 1,
        config: dict = {},
    ) -> None:
        """
        Initialize the CBLoss.

        Args:
            num_classes (int, optional): Number of target classes.
            reduction (str, optional): Reduction method for the loss.
            alpha (float, optional): Weight in joint training.
            config (dict, optional): Configuration dictionary.
        """
        super(CBLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha if config.training_mode == "joint" else 1.0
        self.concept_loss = config.concept_loss
        assert self.concept_loss in ["bce", "ce"], "concept_loss must be either 'bce' or 'ce'"
        # self.reduction = reduction

    def forward(
        self,
        concepts_pred_probs: Tensor,
        concepts_true: Tensor,
        target_pred_logits: Tensor,
        target_true: Tensor,
        concept_pred_logits: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the loss.

        Args:
            concepts_pred_probs (Tensor): Predicted concept probabilities.
            concepts_true (Tensor): Ground-truth concept values.
            target_pred_logits (Tensor): Predicted target logits.
            target_true (Tensor): Ground-truth target values.

        Returns:
            Tensor: Target loss, concept loss, and total loss.
        """

        concepts_loss = 0

        assert torch.all((concepts_true == 0) | (concepts_true == 1))

        per_concept_loss = []
        if self.concept_loss == "bce":
            for concept_idx in range(concepts_true.shape[1]):
                per_c_loss = F.binary_cross_entropy(
                    concepts_pred_probs[:, concept_idx],
                    concepts_true[:, concept_idx].float(),
                    reduction="none",
                )
                per_concept_loss.append(per_c_loss)
                c_loss = per_c_loss.mean()
                concepts_loss += c_loss
            per_concept_loss = torch.stack(per_concept_loss, dim=1)
        
        elif self.concept_loss == 'ce':
            # Assuming concepts_true is one-hot encoded
            concept_true_labels = concepts_true.argmax(dim=-1)
            per_concept_loss = F.cross_entropy(
                concept_pred_logits, concept_true_labels.long(), reduction="none"
            )
            concepts_loss = per_concept_loss.mean()
        
        if self.num_classes == 2:
            # Logits to probs
            target_pred_probs = nn.Sigmoid()(target_pred_logits.squeeze(1))
            per_sample_target_loss = F.binary_cross_entropy(
                target_pred_probs, target_true.float(), reduction="none"
            )
        else:
            per_sample_target_loss = F.cross_entropy(
                target_pred_logits, target_true.long(), reduction="none"
            )
        target_loss = per_sample_target_loss.mean()
        total_loss = target_loss + self.alpha * concepts_loss

        loss_dict = {
            "target_loss": target_loss,
            "concepts_loss": concepts_loss,
            "total_loss": total_loss,
            "per_concept_loss": per_concept_loss,
            "per_sample_target_loss": per_sample_target_loss,
        }
        # per_concept_loss, per_sample_target_loss = None, None
        return loss_dict


class ConceptErrorMixtureLoss(nn.Module):

    def __init__(self, config, concept_loss='bce', warmup=0):
        """
        Initialize the ConceptErrorMixtureLoss.

        Args:
            n_clusters (int): Number of clusters.
            n_classes (int): Number of classes.
        """
        super(ConceptErrorMixtureLoss, self).__init__()
        self.config = config
        # self.lambda_c = config.lambda_c
        self.lambda_c1 = config.lambda_c1
        self.lambda_c2 = config.lambda_c2
        self.lambda_t = config.lambda_t

        self.concept_loss = concept_loss
        self.use_concept_loss = config.use_concept_loss
        self.use_task_loss = config.use_task_loss
        self.use_gmm_loss = config.use_gmm_loss

        self.gmm_loss = GaussianMixtureLoss()

    @staticmethod
    def _task_loss(y_true, y_logits, y_true_logits, y_pred_logits):
        if y_true_logits.shape[-1] == 1:
            # Logits to probs
            y_preds = torch.sigmoid(y_logits.squeeze(1))
            y_true_probs = torch.sigmoid(y_true_logits.squeeze(1))
            y_pred_probs = torch.sigmoid(y_pred_logits.squeeze(1))

            target_gt_loss = F.binary_cross_entropy(
                y_true_probs, y_true.float(), reduction="mean"
            )
            target_pred_loss = F.binary_cross_entropy(
                y_pred_probs, y_preds.float(), reduction="mean"
            )
        else:
            target_gt_loss = F.cross_entropy(
                y_true_logits, y_true.long(), reduction="mean"
            )
            y_preds = torch.softmax(y_logits, dim=1)
            target_pred_loss = F.cross_entropy(
                y_pred_logits, y_preds.long(), reduction="mean"
            )
        return target_gt_loss, target_pred_loss

    def _concept_loss(self, concepts_gt, concepts_logits, concepts_preds, concepts_pred_logits):
        concept_gt_loss = torch.tensor(0.0, device=concepts_logits.device)
        concept_pred_loss = torch.tensor(0.0, device=concepts_logits.device)
        
        if self.concept_loss == 'bce':
            # Assuming concepts_gt is binary (0 or 1)
            concepts_pred_probs = torch.sigmoid(concepts_logits)
            for concept_idx in range(concepts_gt.shape[1]):
                c_loss = F.binary_cross_entropy(
                    concepts_pred_probs[:, concept_idx],
                    concepts_gt[:, concept_idx].float(),
                    reduction="mean",
                )
                concept_gt_loss += c_loss

            concepts_pred_probs = torch.sigmoid(concepts_pred_logits)
            for concept_idx in range(concepts_gt.shape[1]):
                c_loss = F.binary_cross_entropy(
                    concepts_pred_probs[:, concept_idx],
                    concepts_preds[:, concept_idx].float(),
                    reduction="mean",
                )
                concept_pred_loss += c_loss
        elif self.concept_loss == 'ce':
            # Assuming concepts_true is one-hot encoded
            concepts_gt = concepts_gt.argmax(dim=-1)
            concept_gt_loss = F.cross_entropy(
                concepts_logits, concepts_gt.long(), reduction="mean"
            )
            concepts_preds = concepts_preds.argmax(dim=-1)
            concept_pred_loss = F.cross_entropy(
                concepts_pred_logits, concepts_preds.long(), reduction="mean"
            )
            
        return concept_gt_loss, concept_pred_loss
    
    def forward(
            self,
            pred_log_likelihood_matrix: Tensor,
            concepts_gt: Tensor,
            concepts_logits: Tensor,
            concepts_preds: Tensor,
            concepts_pred_logits: Tensor,
            y_true: Optional[Tensor] = None,
            y_logits: Optional[Tensor] = None,
            y_true_logits: Optional[Tensor] = None,
            y_pred_logits: Optional[Tensor] = None,
        ):
        
        concept_gt_loss = torch.tensor(0.0, device=concepts_logits.device)
        concept_pred_loss = torch.tensor(0.0, device=concepts_logits.device)
        if self.use_concept_loss:
            # Compute concept losses
            concept_gt_loss, concept_pred_loss = self._concept_loss(
                concepts_gt, concepts_logits, concepts_preds, concepts_pred_logits
            )
        # Compute Target loss if available
        target_gt_loss = torch.tensor(0.0, device=concepts_logits.device)
        target_pred_loss = torch.tensor(0.0, device=concepts_logits.device)
        if self.use_task_loss:
            target_gt_loss, target_pred_loss = self._task_loss(
                y_true, y_logits, y_true_logits, y_pred_logits
            )
        # Compute the GMM loss
        mixture_loss = torch.tensor(0.0, device=concepts_logits.device)
        if self.use_gmm_loss:
            mixture_loss = self.gmm_loss(pred_log_likelihood_matrix)
        # Compute the total loss
        total_loss = mixture_loss + \
            (self.lambda_c1 * concept_gt_loss + self.lambda_c2 * concept_pred_loss) + \
            self.lambda_t * (target_gt_loss + target_pred_loss)

        loss_dict = {
            "concept_gt_loss": concept_gt_loss,
            "concept_pred_loss": concept_pred_loss,
            "mixture_loss": mixture_loss,
            "target_gt_loss": target_gt_loss,
            "target_pred_loss": target_pred_loss,
            "total_loss": total_loss,
        }

        return loss_dict


class GaussianMixtureLoss(nn.Module):
    """
    Loss function for the Gaussian Mixture Model (GMM).
    """

    def forward(self, s_k, uncertainty=None):
        """
        Compute the loss.

        Args:
            log_likelihood_matrix (Tensor): Log-likelihood matrix from GMM.
            y (Tensor): Ground-truth target values.
            y_pred (Tensor): Predicted target values.
            uncertainty (Tensor, optional): Uncertainty estimates.

        Returns:
            Tensor: The computed loss.
        """
        _, n_components = s_k.shape
        s_max = torch.max(s_k, dim=-1)[0]
        s_max_rep = s_max.unsqueeze(1).repeat(1, n_components)
        l_f = torch.logsumexp(s_k - s_max_rep, dim=1)
        log_prob = s_max + l_f
        # sum_log_prob = torch.sum(log_prob, dim=1)
        loss = -torch.mean(log_prob)

        return loss
    