import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_norm(x):
    return torch.norm(x, p=2, dim=-1)

def rbf_gaussian(x):
    return torch.exp(-x.pow(2))

class RBFLayer(nn.Module):
    def __init__(self, in_features_dim, num_kernels, out_features_dim, 
                 radial_function=rbf_gaussian, norm_function=euclidean_norm, normalization=False):
        super(RBFLayer, self).__init__()
        self.in_features_dim = in_features_dim
        self.num_kernels = num_kernels
        self.out_features_dim = out_features_dim
        self.centers = nn.Parameter(torch.Tensor(num_kernels, in_features_dim))
        self.shapes = nn.Parameter(torch.Tensor(num_kernels))
        self.linear = nn.Linear(num_kernels, out_features_dim)
        self.radial_function = radial_function
        self.norm_function = norm_function
        self.normalization = normalization
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.shapes, 1)

    def forward(self, x):
        size = (x.size(0), self.num_kernels, self.in_features_dim)
        x = x.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        dist = self.norm_function(x - c)
        phi = self.radial_function(dist * self.shapes.unsqueeze(0))
        
        if self.normalization:
            phi = phi / torch.sum(phi, dim=1, keepdim=True)
            
        output = self.linear(phi)
        return output

    def get_kernels_centers(self):
        return self.centers.detach().cpu().numpy()

    def get_shapes(self):
        return self.shapes.detach().cpu().numpy()
