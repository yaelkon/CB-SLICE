"""
GMM and baseline models.
"""

import os
import math
import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, MultivariateNormal
import torch.nn.functional as F
from torchvision import models

from models.networks import FCNNEncoder
from utils.training import freeze_module, unfreeze_module


def create_model(config):
    """
    Parse the configuration file and return a relevant model
    """
    if config.model.model == "cbm":
        return CBM(config)
    elif config.model.model == "dnn":
        return DNN(config)
    else:
        print("Could not create model with name ", config.model, "!")
        quit()


class DNN(nn.Module):
    """
    Deep Neural Network (DNN)
    """
    def __init__(self, config):
        super(DNN, self).__init__()
        self.model = config.model.model
        self.num_classes = config.data.num_classes
        self.encoder_arch = config.model.encoder_arch

        if self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        config.model.model_directory, "resnet/resnet18-5c106cde.pth"
                    ),
                    weights_only=False,
                )
            )
            n_features = self.encoder_res.fc.in_features
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(128, n_features),
                # nn.ReLU(),
            )

        else:
            raise NotImplementedError("ERROR: architecture not supported!")
        
        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes
        else:
            raise NotImplementedError("ERROR: number of classes not supported!")
        
        self.head = nn.Linear(n_features, self.pred_dim)

    def forward(self, x):
        embeddings = self.encoder(x)
        logits = self.head(embeddings)

        return {
            "embeddings": embeddings,
            "logits": logits,
        }


class CBM(nn.Module):
    """
    Model class encompassing all baselines: Hard & Soft Concept Bottleneck Model (CBM),
                                            Concept Embedding Model (CEM), and Autoregressive CBM (AR).

    This class implements the baselines. Depending on the choice of model, only a small part of the full code is used.
    Check the if statements in the forward method to see which part of the code is used for which model.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        concept_learning (str): The concept learning method ("hard", "soft", "embedding", or "autoregressive").
                                This determines the type of method to use
    """

    def __init__(self, config):
        super(CBM, self).__init__()

        # Configuration arguments
        config_model = config.model
        self.config = config_model
        self.num_concepts = config.data.num_concepts
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode
        self.concept_learning = config_model.concept_learning
        if self.concept_learning in ("hard", "autoregressive"):
            if self.training_mode == "joint":
                self.num_epochs = config_model.j_epochs
            else:
                self.num_epochs = config_model.t_epochs
        elif self.concept_learning == "embedding":
            self.CEM_embedding = config_model.embedding_size

        # Architectures
        # Encoder h(.)
        if self.encoder_arch == "FCNN":
            n_features = 256
            self.encoder = FCNNEncoder(
                num_inputs=config.data.num_covariates, num_hidden=n_features, num_deep=2
            )
        elif self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        config_model.model_directory, "resnet/resnet18-5c106cde.pth"
                    ),
                    weights_only=False,
                )
            )
            n_features = self.encoder_res.fc.in_features
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(128, n_features),
                nn.ReLU(),
            )

        else:
            raise NotImplementedError("ERROR: architecture not supported!")
        if self.concept_learning == "embedding":
            print(
                "Please be aware that our implementation of CEMs is without training on interventions! This is because we would deem this an unfair comparison to our method that is also not trained on interventions. Still, be careful when using this CEM code for derivative works"
            )
            self.positive_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(n_features, self.CEM_embedding, bias=True),
                        nn.LeakyReLU(),
                    )
                    for _ in range(self.num_concepts)
                ]
            )
            self.negative_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(n_features, self.CEM_embedding, bias=True),
                        nn.LeakyReLU(),
                    )
                    for _ in range(self.num_concepts)
                ]
            )
            self.scoring_function = nn.Sequential(
                nn.Linear(self.CEM_embedding * 2, 1, bias=True), nn.Sigmoid()
            )
            self.concept_dim = self.CEM_embedding * self.num_concepts
        else:
            if self.concept_learning == "autoregressive":
                self.concept_predictor = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(n_features + i, 50, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(50, 1, bias=True),
                        )
                        for i in range(self.num_concepts)
                    ]
                )

            else:
                self.concept_predictor = nn.Linear(
                    n_features, self.num_concepts, bias=True
                )
            self.concept_dim = self.num_concepts

        # Assume binary concepts
        assert config_model.concept_loss in ("bce", "ce"), "concept_loss must be either 'bce' or 'ce'"
        self.act_c = nn.Sigmoid() if config_model.concept_loss == "bce" else nn.Softmax(dim=-1)

        # Link function g(.)
        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes

        if self.head_arch == "linear":
            fc_y = nn.Linear(self.concept_dim, self.pred_dim)
            self.head = nn.Sequential(fc_y)
        else:
            fc1_y = nn.Linear(self.concept_dim, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)

    def forward(
            self,
            x,
            validation=False,
    ):
        """
        Perform a forward pass through one of the baselines.

        This method performs a forward pass predicting concept probabilities and logits for the target variable.
        It handles different concept learning strategies and training modes, including hard, soft, autoregressive, and embedding-based concepts.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            concepts_train_ar (torch.Tensor, optional): Ground-truth concept values for autoregressive training. Default is False.

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, label_dim)
                - c (torch.Tensor): Predicted hard concept values (if method permits, otherwise the concept representation). Shape: (batch_size, num_concepts, num_monte_carlo) for MCMC sampling or (batch_size, num_concepts) otherwise.
        """

        # Get intermediate representations
        intermediate = self.encoder(x)

        # Get concept predictions
        if self.concept_learning in ("hard", "soft"):
            # CBM
            c_logit = self.concept_predictor(intermediate)
            c_prob = self.act_c(c_logit)

            if self.concept_learning in ("hard"):
                # Hard CBM
                if self.config.concept_loss == "bce":
                    c = (c_prob > 0.5).float()
                elif self.config.concept_loss == "ce":
                    c = torch.argmax(c_prob, dim=-1)
                    # Convert to one-hot encoding
                    c = F.one_hot(c.long(), num_classes=self.num_concepts).float()
            else:
                c = c_prob

        elif self.concept_learning == "embedding":
            # CEM
            if self.training_mode == "joint":
                # Obtaining concept embeddings
                c_p = [p(intermediate) for p in self.positive_embeddings]
                c_n = [n(intermediate) for n in self.negative_embeddings]

                # Concept probabilities from scoring function
                c_prob = [
                    self.scoring_function(torch.cat((c_p[i], c_n[i]), dim=1))
                    for i in range(self.num_concepts)
                ]

                # Final concept embedding
                z_prob = [
                    c_prob[i] * c_p[i] + (1 - c_prob[i]) * c_n[i]
                    for i in range(self.num_concepts)
                ]
                z_prob = torch.cat([z_prob[i] for i in range(self.num_concepts)], dim=1)
                c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
                c = z_prob
            else:
                raise Exception("CEMs are trained jointly, change training mode")

        # Get predicted targets
        if self.concept_learning == "hard":
            # Hard CBM.
            y_pred_logits = self.head(c)

        elif self.concept_learning == "soft":
            # Soft CBM
            y_pred_logits = self.head(c_logit)  # NOTE that we're passing logits not probs in soft case as is also done by Koh et al.
            # c = torch.empty_like(c_prob)

        elif self.concept_learning == "embedding":
            # CEM or training of AR. Takes ground truth concepts.
            # If CEM: c are predicte embeddings, if AR: c are ground truth concepts
            y_pred_logits = self.head(c)

        output = {
            "c_prob": c_prob,
            "y_pred_logits": y_pred_logits,
            "c": c,
            "c_logits": c_logit,
        }
        return output

    def calculate_ectp(self, x) -> torch.Tensor:
        """
        Calculate the Expected Change in Target Prediction for concept intervention.
        """
        eps = 1e-6
        outputs = self(x, validation=True)
        if self.concept_learning == "hard":
            c_out = outputs["c"]
            # c_out: [batch, n_concepts]
            # c_rep: [batch, n_concepts, 2, n_concepts], each slice along dim=1 (concept) contains two identical copies of c_prob as an extra dimension
            c_rep = c_out.tile(2 * self.num_concepts)
            c_rep = c_rep.reshape(c_out.shape[0], self.num_concepts, 2, self.num_concepts)
            c_rep[:, torch.arange(self.num_concepts), 0, torch.arange(self.num_concepts)] = 0
            c_rep[:, torch.arange(self.num_concepts), 1, torch.arange(self.num_concepts)] = 1
        
        elif self.concept_learning == "soft":
            c_out = outputs["c_logits"]
            # c_out: [batch, n_concepts]
            # c_rep: [batch, n_concepts, 2, n_concepts], each slice along dim=1 (concept) contains two identical copies of c_prob as an extra dimension
            c_rep = c_out.tile(2 * self.num_concepts)
            c_rep = c_rep.reshape(c_out.shape[0], self.num_concepts, 2, self.num_concepts)
            c_rep[:, torch.arange(self.num_concepts), 0, torch.arange(self.num_concepts)] = torch.logit(torch.tensor([eps], device=c_out.device))
            c_rep[:, torch.arange(self.num_concepts), 1, torch.arange(self.num_concepts)] = torch.logit(torch.tensor([1 - eps], device=c_out.device))
        
        c_neg_interv = c_rep[:, :, 0]  # [batch, n_concepts, n_concepts]
        c_pos_interv = c_rep[:, :, 1]  # [batch, n_concepts, n_concepts]

        # Get original prediction (baseline)
        y_logits_orig = outputs["y_pred_logits"]  # [batch, pred_dim]
        
        # Reshape interventions to process all in one batch
        batch_size = c_neg_interv.shape[0]
        n_concepts = c_neg_interv.shape[1]
        
        # Reshape: [batch, n_concepts, n_concepts] -> [batch * n_concepts, n_concepts]
        c_neg_interv_flat = c_neg_interv.reshape(batch_size * n_concepts, n_concepts)
        c_pos_interv_flat = c_pos_interv.reshape(batch_size * n_concepts, n_concepts)
        
        # Forward through head in batched mode
        y_logits_neg = self.head(c_neg_interv_flat)  # [batch * n_concepts, pred_dim]
        y_logits_pos = self.head(c_pos_interv_flat)  # [batch * n_concepts, pred_dim]
        
        if y_logits_orig.size(1) == 1:  # Binary classification
            y_pred_orig = torch.sigmoid(y_logits_orig)
            y_pred_neg = torch.sigmoid(y_logits_neg)
            y_pred_pos = torch.sigmoid(y_logits_pos)
        else:  # Multi-class classification
            y_pred_orig = torch.softmax(y_logits_orig, dim=-1)
            y_pred_neg = torch.softmax(y_logits_neg, dim=-1)
            y_pred_pos = torch.softmax(y_logits_pos, dim=-1)

        # Reshape back: [batch * n_concepts, pred_dim] -> [batch, n_concepts, pred_dim]
        y_pred_neg = y_pred_neg.reshape(batch_size, n_concepts, -1)
        y_pred_pos = y_pred_pos.reshape(batch_size, n_concepts, -1)
        
        # Calculate Expected Change in Target Prediction (ECTP)
        # Clamp probabilities for numerical stability
        y_pred_neg_safe = torch.clamp(y_pred_neg, eps, 1 - eps)
        y_pred_pos_safe = torch.clamp(y_pred_pos, eps, 1 - eps)
        y_pred_orig_safe = torch.clamp(y_pred_orig, eps, 1 - eps)

        # Binary classification - use Bernoulli KL divergence
        if y_logits_orig.size(1) == 1:
            # D_KL(P || Q) = P*log(P/Q) + (1-P)*log((1-P)/(1-Q))
            y_pred_orig_expanded = y_pred_orig_safe.unsqueeze(1)

            # KL divergence for setting concept to 0
            kl_neg_term1 = y_pred_neg_safe * (
                torch.log(y_pred_neg_safe) -
                torch.log(y_pred_orig_expanded)
            )
            kl_neg_term1 = torch.max(torch.zeros_like(kl_neg_term1), kl_neg_term1)

            kl_neg_term2 = (1 - y_pred_neg_safe) * (
                torch.log(1 - y_pred_neg_safe) -
                torch.log(1 - y_pred_orig_expanded)
            )
            kl_neg_term2 = torch.max(torch.zeros_like(kl_neg_term2), kl_neg_term2)

            kl_neg = (kl_neg_term1 + kl_neg_term2).squeeze(-1)

            # KL divergence for setting concept to 1
            kl_pos_term1 = y_pred_pos_safe * (
                torch.log(y_pred_pos_safe) -
                torch.log(y_pred_orig_expanded)
            )
            kl_pos_term1 = torch.max(torch.zeros_like(kl_pos_term1), kl_pos_term1)

            kl_pos_term2 = (1 - y_pred_pos_safe) * (
                torch.log(1 - y_pred_pos_safe) -
                torch.log(1 - y_pred_orig_expanded)
            )
            kl_pos_term2 = torch.max(torch.zeros_like(kl_pos_term2), kl_pos_term2)

            kl_pos = (kl_pos_term1 + kl_pos_term2).squeeze(-1)
        else:  # Multi-class classification
            # D_KL(P || Q) = Σ_i P_i * log(P_i / Q_i)
            y_pred_orig_expanded = y_pred_orig_safe.unsqueeze(1)

            # KL divergence for setting concept to 0
            kl_neg = y_pred_neg_safe * (
                torch.log(y_pred_neg_safe) -
                torch.log(y_pred_orig_expanded)
            )
            kl_neg = kl_neg.sum(dim=-1)
            kl_neg = torch.max(torch.zeros_like(kl_neg), kl_neg)

            # KL divergence for setting concept to 1
            kl_pos = y_pred_pos_safe * (
                torch.log(y_pred_pos_safe) -
                torch.log(y_pred_orig_expanded)
            )
            kl_pos = kl_pos.sum(dim=-1)
            kl_pos = torch.max(torch.zeros_like(kl_pos), kl_pos)
        
        # ECTP across both intervention directions
        c_probs = outputs["c_prob"]
        ectp_score = (1 - c_probs) * kl_neg + c_probs * kl_pos  # [batch, n_concepts]
        
        return ectp_score

    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.concept_predictor.apply(freeze_module)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


from .layers import MixtureGaussianLayer


class MixtureGaussiansCBM(nn.Module):
    """
    Mixture of Gaussians Concept Bottleneck Model (CBM).
    This class implements gmm layer for the concpet bottleneck representation.
    Given a set of concept logits, it predicts the parameters of a mixture of gaussians.
    """

    def __init__(
            self,
            pretrained_model,
            n_clusters,
            n_concepts,
            n_features=None,
            n_classes=None,
            use_task_heads=False,
            clustering_method="gmm",
            filtered_concepts=None,
            ):

        assert filtered_concepts is None or len(filtered_concepts) == n_features, "Filtered concepts must be the same length as the number of features"
        
        super(MixtureGaussiansCBM, self).__init__()
        
        self.model = pretrained_model.eval()
        self.n_clusters = n_clusters
        self.n_classes = n_classes
        self.use_task_heads = use_task_heads
        self.clustering_method = clustering_method

        if filtered_concepts is not None:
            self.n_concepts = len(filtered_concepts)
            self.filtered_concepts = torch.tensor(filtered_concepts)
        else:
            self.n_concepts = n_concepts
            self.filtered_concepts = None
        
        self.n_features = n_features
        if n_features is None:
            self.n_features = self.n_concepts

        self.pred_dim = self.n_classes
        if self.n_classes == 2:
            self.pred_dim = 1
            
        # Mixture of Gaussians layer
        if self.clustering_method == "gmm":
            self.clustering_layer = MixtureGaussianLayer(n_gaussians=self.n_clusters, n_features=self.n_features, n_classes=self.n_classes)
        elif self.clustering_method == "linear":
            self.clustering_layer = nn.Linear(self.n_concepts, self.n_clusters)

        self.fc_head_1 = nn.Linear(self.n_clusters, self.n_concepts)
        self.fc_head_2 = nn.Linear(self.n_clusters, self.n_concepts)

        self.fc_head_y1 = None
        self.fc_head_y2 = None
        if self.use_task_heads:
            # If n_classes is specified, add task classification heads
            self.fc_head_y1 = nn.Linear(self.n_clusters, self.pred_dim)
            self.fc_head_y2 = nn.Linear(self.n_clusters, self.pred_dim)
        # self.reducer = TSNE(n_components=2, perplexity=30, random_state=42, verbose=0)

    def init_params(self, x, labels):
        if isinstance(self.clustering_layer, MixtureGaussianLayer):
            self.clustering_layer.init_params(x, labels)
    
    def _filter_concepts(self, cbm_outputs):
        cbm_outputs["c_prob"] = cbm_outputs["c_prob"][:, self.filtered_concepts]
        cbm_outputs["c"] = cbm_outputs["c"][:, self.filtered_concepts]
        cbm_outputs["c_logits"] = cbm_outputs["c_logits"][:, self.filtered_concepts]
        return cbm_outputs
    
    def forward(self, x, labels=None, cbm_validation=True):
        self.model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            cbm_outputs = self.model(x, cbm_validation)
            
            if self.filtered_concepts is not None:
                cbm_outputs = self._filter_concepts(cbm_outputs)
            c_logits = cbm_outputs["c_logits"]  # Get the concept logits from the pretrained model
            
        if self.clustering_method == "linear":
            x = self.clustering_layer(c_logits)
            assignment_probs = torch.softmax(x, dim=-1)
            ll_matrix = None
        elif self.clustering_method == "gmm":
            # Map concept logits to features
            # predict the log likelihood for each example to be assosicated with each cluster
            ll_matrix = self.clustering_layer(c_logits, labels=labels)
            # ll_matrix is the joint probability of the data and the cluster p(x, k)
            # to calculate the posterior p(k|x), we need to normalize it
            ll_max_k = torch.max(ll_matrix, dim=1, keepdim=True)[0]
            assignment_probs = torch.softmax(ll_matrix - ll_max_k, dim=1)
        else:
            raise NotImplementedError(f"Clustering method {self.clustering_method} is not supported")
        
        c_hat_logits = self.fc_head_1(assignment_probs)    
        c_pred_hat_logits = self.fc_head_2(assignment_probs)

        y_pred_logits_1 = None
        y_pred_logits_2 = None
        if self.use_task_heads:
            y_pred_logits_1 = self.fc_head_y1(assignment_probs)
            y_pred_logits_2 = self.fc_head_y2(assignment_probs)

        outputs = {
            "log_lokelihood_matrix": ll_matrix,
            "assignment_probs": assignment_probs,
            "c_true_logits": c_hat_logits,
            "c_pred_logits": c_pred_hat_logits,
            "y_pred_logits_1": y_pred_logits_1,
            "y_pred_logits_2": y_pred_logits_2,
            "cbm_outputs": cbm_outputs,
        }

        return outputs

    def calculate_ecca(self, x):
        """
        Calculate the expected change in concept assignment (ECCA) for concept intervention.
        """
        eps = 1e-6
        outputs = self(x)

        c_logits = outputs["cbm_outputs"]["c_logits"]
        c_rep = c_logits.tile(2 * self.n_concepts)
        c_rep = c_rep.reshape(c_logits.shape[0], self.n_concepts, 2, self.n_concepts)
        c_rep[:, torch.arange(self.n_concepts), 0, torch.arange(self.n_concepts)] = torch.logit(torch.tensor([eps], device=c_logits.device))
        c_rep[:, torch.arange(self.n_concepts), 1, torch.arange(self.n_concepts)] = torch.logit(torch.tensor([1 - eps], device=c_logits.device))        

        c_neg_interv = c_rep[:, :, 0]  # [batch, n_concepts, n_concepts]
        c_pos_interv = c_rep[:, :, 1]  # [batch, n_concepts, n_concepts]

        # Get original assignment probabilities
        assignment_probs_orig = outputs["assignment_probs"]
        # Reshape interventions to process all in one batch
        batch_size = c_neg_interv.shape[0]
        n_concepts = c_neg_interv.shape[1]
        
        # Reshape: [batch, n_concepts, n_concepts] -> [batch * n_concepts, n_concepts]
        c_neg_interv_flat = c_neg_interv.reshape(batch_size * n_concepts, n_concepts)
        c_pos_interv_flat = c_pos_interv.reshape(batch_size * n_concepts, n_concepts)
    
        ll_matrix_neg = self.clustering_layer(c_neg_interv_flat)
        ll_matrix_pos = self.clustering_layer(c_pos_interv_flat)
        # ll_matrix is the joint probability of the data and the cluster p(x, k)
        # to calculate the posterior p(k|x), we need to normalize it
        ll_max_k_neg = torch.max(ll_matrix_neg, dim=1, keepdim=True)[0]
        ll_max_k_pos = torch.max(ll_matrix_pos, dim=1, keepdim=True)[0]
        assignment_probs_neg = torch.softmax(ll_matrix_neg - ll_max_k_neg, dim=1)
        assignment_probs_pos = torch.softmax(ll_matrix_pos - ll_max_k_pos, dim=1)

        # Reshape back: [batch * n_concepts, n_clusters] -> [batch, n_concepts, n_clusters]
        assignment_probs_neg = assignment_probs_neg.reshape(batch_size, n_concepts, -1)
        assignment_probs_pos = assignment_probs_pos.reshape(batch_size, n_concepts, -1)
        
        # Calculate Expected Change in Concept Assignment (ECCA)
        # Clamp probabilities for numerical stability
        assignment_probs_neg_safe = torch.clamp(assignment_probs_neg, eps, 1 - eps)
        assignment_probs_pos_safe = torch.clamp(assignment_probs_pos, eps, 1 - eps)
        assignment_probs_orig_safe = torch.clamp(assignment_probs_orig, eps, 1 - eps)
        
        # Calculating multi-class KL divergence
        assignment_probs_orig_expanded = assignment_probs_orig_safe.unsqueeze(1)

        # KL divergence for setting concept to 0
        kl_neg = assignment_probs_neg_safe * (
            torch.log(assignment_probs_neg_safe) -
            torch.log(assignment_probs_orig_expanded)
        )
        kl_neg = kl_neg.sum(dim=-1)
        kl_neg = torch.max(torch.zeros_like(kl_neg), kl_neg)

        # KL divergence for setting concept to 1
        kl_pos = assignment_probs_pos_safe * (
            torch.log(assignment_probs_pos_safe) -
            torch.log(assignment_probs_orig_expanded)
        )
        kl_pos = kl_pos.sum(dim=-1)
        kl_pos = torch.max(torch.zeros_like(kl_pos), kl_pos)

        c_probs = outputs["cbm_outputs"]["c_prob"]
        ecca_score = (1 - c_probs) * kl_neg + c_probs * kl_pos  # [batch, n_concepts]
        return ecca_score

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        self.clustering_layer._init_params = True
