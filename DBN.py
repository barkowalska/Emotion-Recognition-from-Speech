import torch
import numpy as np
from RBM import *

class DBN:
    def __init__(self, input_size, hidden_shape, cd_k, stddev):
        self.input_size = input_size
        self.hidden_shape = hidden_shape
        self.cd_k = cd_k
        self.stddev = stddev
        self.layer_parameters = [{'W': None, 'bp': None, 'bn': None, 'stddev': None} for _ in range(len(hidden_shape))]

    def train(self, data, epochs, batch_size, learning_rate, momentum):
        data = torch.tensor(data, dtype=torch.float32)
        
        for index in range(len(self.layer_parameters)):
            if index == 0:
                vn = self.input_size
            else:
                vn = self.hidden_shape[index - 1]
            hn = self.hidden_shape[index]
            
            if index == 0:
                gb_rbm = GB_RBM(vn, hn, self.cd_k, self.stddev)
                gb_rbm.train(data, epochs, batch_size, learning_rate, momentum)
                self.layer_parameters[index]['W'] = gb_rbm.weights.clone().detach().float()
                self.layer_parameters[index]['bp'] = gb_rbm.bias_p.clone().detach().float()
                self.layer_parameters[index]['bn'] = gb_rbm.bias_n.clone().detach().float()
                self.layer_parameters[index]['stddev'] = gb_rbm.v_stddev.clone().detach().float()
            else:
                rbm = RBM(vn, hn, self.cd_k)
                data_dash = self.generate_input_for_layer(index, data)
                rbm.train(data_dash, epochs, batch_size, learning_rate, momentum)
                self.layer_parameters[index]['W'] = rbm.weights.clone().detach().float()
                self.layer_parameters[index]['bp'] = rbm.bias_p.clone().detach().float()
                self.layer_parameters[index]['bn'] = rbm.bias_n.clone().detach().float()

            print("Finished Training Layer:", index, "to", index + 1)

    def forward_pass(self, layer, input):
        input = input.float()
        W = self.layer_parameters[layer]['W']
        bp = self.layer_parameters[layer]['bp']
        if layer == 0:
            stddev = self.layer_parameters[layer]['stddev']
            activation = (input @ W) / (stddev ** 2) + bp
        else:
            activation = (input @ W) + bp
        probability = torch.sigmoid(activation)
        h = torch.bernoulli(probability)
        return probability, h

    def backward_pass(self, layer, hidden):
        W = self.layer_parameters[layer]['W']
        bn = self.layer_parameters[layer]['bn']
        activation = hidden @ W.T + bn
        if layer == 0:
            stddev = self.layer_parameters[layer]['stddev']
            v = torch.normal(mean=activation, std=stddev)
        else:
            probability = torch.sigmoid(activation)
            v = torch.bernoulli(probability)
        return v

    def evaluate(self, dataset):
        error = torch.zeros(dataset.size(1))
        for idx in range(dataset.size(0)):
            error += (dataset[idx, :] - self.reconstruct(dataset[idx, :]))**2
        mean_error = torch.mean(error)/dataset.size(0)
        print(f"Evaluation error: {mean_error.item()}")
        return mean_error.item()
    
    
    def generate_input_for_layer(self, index, data):
        x_dash = data.clone()
        for i in range(index):
            _, x_dash = self.forward_pass(i, x_dash)
        return x_dash

    def reconstruct(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.float32)
        visible = self.generate_input_for_layer(len(self.layer_parameters), observation)
        layers = len(self.layer_parameters)
        layers = int(layers)
        for index in range(0, layers):
            visible = self.backward_pass(layers - 1 - index, visible)
        return visible

    def save_model(self, file_path):
        torch.save(self.layer_parameters, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.layer_parameters = torch.load(file_path)
        print(f"Model loaded from {file_path}")
