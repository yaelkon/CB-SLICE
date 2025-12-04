import torch
import math
import numpy as np

from torch import nn
from torch.distributions import MultivariateNormal

class MixtureGaussianLayer(nn.Module):
    """
    Gaussian Mixture Model Layer
    """

    def __init__(self, n_gaussians, n_features, n_classes=None):
        super(MixtureGaussianLayer, self).__init__()
        self.n_features = n_features
        self.n_gaussians = n_gaussians
        self.n_classes = n_classes
        self._num_g_per_class = int(self.n_gaussians // self.n_classes)
        
        # Define the parameters for the Gaussian Mixture Model
        self.means = nn.Parameter(torch.randn(n_gaussians, n_features), requires_grad=True)
        self.vars = nn.Parameter(torch.ones(n_gaussians, n_features), requires_grad=True)
        self.mixture_weights = nn.Parameter(torch.ones(n_gaussians,), requires_grad=True)
        
        self.eps = 1e-4
        self._init_params = False

    def init_params(self, x, labels=None):
        print(f"Initialising Gaussian Mixture Model parameters")
        # Initialize the parameters of the GMM layer
        self.mixture_weights.data.fill_(1.0 / self.n_gaussians)

        if labels is not None:
            n_unique_labels = labels.unique().shape[0]
        else:
            n_unique_labels = -1
        
        # If we have representative examples for each class, use them to initialize the means
        if n_unique_labels == self.n_classes:
        
            num_g_per_class = self._num_g_per_class
            # If labels are provided, use them to initialize means
            for i, label in enumerate(np.arange(0, self.n_classes)):
                if (i * num_g_per_class) <= self.n_gaussians:
                    # Initialize means for each unique label
                    x_mu = x[labels == label].mean(dim=0)
                    if x[labels == label].shape[0] > 1:
                        x_std = torch.diag(x[labels == label].std(dim=0))
                    else:
                        x_std = torch.eye(self.n_features, device=x.device)
                    mu_dist = MultivariateNormal(x_mu, x_std)
                    init_mu = mu_dist.rsample([num_g_per_class])
                    
                    start = i * num_g_per_class
                    end = (i + 1) * num_g_per_class
                    if end > self.n_gaussians:
                        end = self.n_gaussians
                        num_g_per_class = end - start
                    self.means.data[start : end] = init_mu[:num_g_per_class]

                else:
                    break
        else:
            # If we don't have representative examples for each class, use all examples together
            print(f"Number of unique labels {n_unique_labels} does not match the number of classes {self.n_classes}")
            print("Using all examples for Gaussian Mixture Model initialisation")
            x_mu = torch.mean(x, dim=0)
            x_std = torch.diag(torch.std(x, dim=0))
            mu_dist = MultivariateNormal(x_mu, x_std)
            init_mu = mu_dist.rsample([self.n_gaussians])
            self.means.data = init_mu
        # self.vars.data = torch.ones_like(self.means)  # Initialize variances to a small value
        self.vars.data = torch.var(x, dim=0).unsqueeze(0).repeat(self.n_gaussians, 1) + self.eps
        
        self._init_params = True
    
    def forward(self, x, labels=None):
        # Implement the forward pass of the GMM layer
        if not self._init_params:
            self.init_params(x, labels)
        
        # Compute the Gaussian distribution function
        # Preventing numerical instability
        means = self.means
        # print(f"min vars: {self.vars.min()}, max vars: {self.vars.max()}")
        clamped_vars = torch.clamp(self.vars, min=self.eps, max=None)
        mixture_weights = torch.softmax(self.mixture_weights, dim=0)
        
        # Preparing matrixs
        batch_size, n_dims = x.size()
        x_rep = x.unsqueeze(1).repeat(1, self.n_gaussians, 1)
        mu_rep = means.unsqueeze(0).repeat(batch_size, 1, 1)
        var_rep = clamped_vars.unsqueeze(0).repeat(batch_size, 1, 1)
        mixture_weights_rep = mixture_weights.unsqueeze(0).repeat(batch_size, 1)

        # Compute Mahalanobis distance
        dist = (1 / 2) * (x_rep - mu_rep) * (1 / var_rep) * (x_rep - mu_rep)
        dist = dist.sum(dim=-1)
        # Compute constants
        cons = (n_dims / 2) * torch.log(torch.tensor((2 * math.pi), device=x.device))
        # Assumimg diagonal covariance matrix, thus its determinant is equal to the product of its diagonal values
        # var_rep = torch.clamp(var_rep, min=self.eps, max=2)
        # var_det = torch.prod(var_rep, dim=-1)
        
        # Compute the coefficient. Add eps to the determinant to avoid numerical instability
        coef = - cons - (1 / 2) * torch.sum(torch.log(var_rep), dim=-1)
        # Compute the log likelihood
        l_p_k = coef - dist
        # Compute the weighted log likelihood
        l_a_k = torch.log(mixture_weights_rep + self.eps)
        # Log of the joint probability \log p(x, k) = \log a_k + \log p_k(x)
        s_k = l_a_k + l_p_k

        return s_k
