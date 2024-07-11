import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import math


class RBM:
    def __init__(self, visible_n, hidden_n, cd_k):
        self.visible_n = visible_n
        self.hidden_n = hidden_n

        # Xavier init
        self.weights = torch.normal(mean=0, std=1.0/(visible_n+hidden_n), size= (visible_n, hidden_n),dtype=torch.double )
        self.bias_p = torch.zeros(hidden_n, dtype=torch.double )
        self.bias_n = torch.zeros(visible_n, dtype=torch.double )

        #Gradients
        self.grad_W = torch.zeros(size= (visible_n, hidden_n))
        self.grad_bp = torch.zeros(hidden_n)
        self.grad_bn = torch.zeros(visible_n)

        #Moments
        self.moment_W = torch.zeros(size= (visible_n, hidden_n))
        self.moment_bp = torch.zeros(hidden_n)
        self.moment_bn = torch.zeros(visible_n)

        self.k = cd_k


    def forward_pass(self, v):
        activation = (v @ self.weights) + self.bias_p
        probablity = torch.sigmoid(activation)
        h = torch.bernoulli(probablity)
        return probablity, h
    
    def backward_pass(self, h):
        activation = h @ self.weights.t() + self.bias_n
        probablity = torch.sigmoid(activation)
        v = torch.bernoulli(probablity)
        return probablity, v

    def contrastive_divergence(self, v):
        probh, h = self.forward_pass(v)
        for _ in range(self.k):
            probv, v = self.backward_pass(h)
            probh, h = self.forward_pass(v)
        return probv, v, probh, h

    def gradient_and_error(self, batch):
        b_size = batch.shape[0]
        v0 = batch
        probh0, h0 = self.forward_pass(v0)
        probv1, v1, probh1, h1 = self.contrastive_divergence(v0)
        self.grad_W = (v0.t() @ probh0 - probv1.t() @ probh1) / b_size
        #self.grad_W /= torch.norm(self.grad_W)

        self.grad_bp = (torch.sum(probh0, dim=0) - torch.sum(probh1, dim=0)) / b_size
        #self.grad_bp /= torch.norm(self.grad_bp)

        self.grad_bn = (torch.sum(v0, dim=0) - torch.sum(v1, dim=0)) / b_size
        #self.grad_bn /= torch.norm(self.grad_bn)

        recon_err = torch.mean(torch.sum((v0 - v1) ** 2, dim=1))  # sum of squared error averaged over the batch
        return recon_err

    def update_parameters(self, lr, momentum=0.0):
        self.moment_W *= momentum
        self.moment_W += (1.0-momentum)*lr*self.grad_W
        self.weights += self.moment_W

        self.moment_bp *= momentum
        self.moment_bp += (1.0-momentum)*lr*self.grad_bp
        self.bias_p += self.moment_bp

        self.moment_bn *= momentum
        self.moment_bn += (1.0-momentum)*lr*self.grad_bn
        self.bias_n += self.moment_bn

    def reconstruct(self, v):
        _, v, _, _ = self.contrastive_divergence(v)
        return v

    def get_batches(self, data, batch_size, shuffle=False):
        if shuffle:
            data = data[torch.randperm(data.size(0))]
        num_batches = math.ceil(data.size(0) / batch_size)
        for batch_num in range(num_batches):
            yield data[batch_num * batch_size:(batch_num + 1) * batch_size]

    def train(self, data, epochs, batch_size, learning_rate, momentum=0.0):
        data = torch.tensor(data, dtype=torch.double)
        epoch_error = 0.0
        for epoch in range(epochs):
            last_epoch_error = epoch_error
            epoch_error = 0.0
            for batch in self.get_batches(data, batch_size, True):
                error = self.gradient_and_error(batch)
                self.update_parameters(learning_rate, momentum)
                epoch_error += error.item()

            print(f"Epoch: {epoch}\t Error: {epoch_error/len(data):.4f}")
            if last_epoch_error != 0:
                if abs(last_epoch_error - epoch_error)/last_epoch_error < 0.005:
                    print(f"No more improvement. End.")
                    break


class GB_RBM:
    def __init__(self, visible_n, hidden_n, cd_k, stddev):
        self.visible_n = visible_n
        self.hidden_n = hidden_n

        # Xavier init
        self.weights = torch.normal(mean=0, std=1.0/(visible_n+hidden_n), size= (visible_n, hidden_n),dtype=torch.double )
        self.bias_p = torch.zeros(hidden_n, dtype=torch.double )
        self.bias_n = torch.zeros(visible_n, dtype=torch.double )

        #Gradients
        self.grad_W = torch.zeros(size= (visible_n, hidden_n))
        self.grad_bp = torch.zeros(hidden_n)
        self.grad_bn = torch.zeros(visible_n)

        #Moments
        self.moment_W = torch.zeros(size= (visible_n, hidden_n))
        self.moment_bp = torch.zeros(hidden_n)
        self.moment_bn = torch.zeros(visible_n)

        self.k = cd_k
        self.v_stddev = torch.tensor(stddev, dtype=torch.double)
        self.grad_v_stddev = torch.tensor(0.0)
        self.v_stddev_moment = torch.tensor(0.0)

    def forward_pass(self, v):
        activation = (v @ self.weights) / (self.v_stddev * self.v_stddev) + self.bias_p
        probablity = torch.sigmoid(activation)
        h = torch.bernoulli(probablity)
        return probablity, h
    
    def backward_pass(self, h):
        activation = h @ self.weights.t() + self.bias_n
        v = torch.normal(mean=activation, std=self.v_stddev)
        return v

    def contrastive_divergence(self, v):
        probh, h = self.forward_pass(v)
        for _ in range(self.k):
            v = self.backward_pass(h)
            probh, h = self.forward_pass(v)
        return v, probh, h

    def gradient_and_error(self, batch):
        b_size = batch.shape[0]
        v0 = batch
        probh0, h0 = self.forward_pass(v0)
        v1, probh1, h1 = self.contrastive_divergence(v0)
        self.grad_W = (v0.t() @ probh0 - v1.t() @ probh1) / (self.v_stddev * self.v_stddev * b_size)
        #self.grad_W /= torch.norm(self.grad_W)

        self.grad_bp = (torch.sum(probh0, dim=0) - torch.sum(probh1, dim=0)) / b_size
        #self.grad_bp /= torch.norm(self.grad_bp)

        self.grad_bn = (torch.sum(v0, dim=0) - torch.sum(v1, dim=0)) / (self.v_stddev * self.v_stddev * b_size)
        #self.grad_bn /= torch.norm(self.grad_bn)

        self.grad_v_stddev = torch.mean((torch.norm(v0 - self.bias_n, dim=1) ** 2 - 2 * torch.matmul(v0, torch.matmul(self.weights, probh0.t())).diag()) / (self.v_stddev ** 3) - 
                                        (torch.norm(v1 - self.bias_n, dim=1) ** 2 - 2 * torch.matmul(v1, torch.matmul(self.weights, probh1.t())).diag()) / (self.v_stddev ** 3))
        
        recon_err = torch.mean(torch.sum((v0 - v1) ** 2, dim=1))  # sum of squared error averaged over the batch
        return recon_err

    def update_parameters(self, lr, momentum=0.0):
        self.moment_W *= momentum
        self.moment_W += (1.0-momentum)*lr*self.grad_W
        self.weights += self.moment_W

        self.moment_bp *= momentum
        self.moment_bp += (1.0-momentum)*lr*self.grad_bp
        self.bias_p += self.moment_bp

        self.moment_bn *= momentum
        self.moment_bn += (1.0-momentum)*lr*self.grad_bn
        self.bias_n += self.moment_bn

        self.v_stddev_moment *= momentum
        self.v_stddev_moment += (1.0-momentum)*lr*self.grad_v_stddev
        if self.v_stddev + self.v_stddev_moment > 0:
            self.v_stddev += self.v_stddev_moment
        #else:
        #    self.v_stddev += self.v_stddev_moment
        #    self.v_stddev = abs(self.v_stddev)

    def reconstruct(self, v):
        v, _, _ = self.contrastive_divergence(v)
        return v

    def get_batches(self, data, batch_size, shuffle=False):
        if shuffle:
            data = data[torch.randperm(data.size(0))]
        num_batches = math.ceil(data.size(0) / batch_size)
        for batch_num in range(num_batches):
            yield data[batch_num * batch_size:(batch_num + 1) * batch_size]

    def train(self, data, epochs, batch_size, learning_rate, momentum=0.0):
        data = torch.tensor(data, dtype=torch.double)
        epoch_error = 0.0
        for epoch in range(epochs):
            last_epoch_error = epoch_error
            epoch_error = 0.0
            for batch in self.get_batches(data, batch_size, True):
                error = self.gradient_and_error(batch)
                self.update_parameters(learning_rate, momentum)
                epoch_error += error.item()

            print(f"Epoch: {epoch}\t Error: {epoch_error/len(data):.4f}")
            if last_epoch_error != 0:
                if abs(last_epoch_error - epoch_error)/last_epoch_error < 0.005:
                    print(f"No more improvement. End.")
                    break

# Example usage:
# rbm = RBM(visible_n=784, hidden_n=256, cd_k=1)
# rbm.train(data, epochs=10, batch_size=64, learning_rate=0.01)
