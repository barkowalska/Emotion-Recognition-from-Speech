from RBM import *

class DBN:
    def __init__ (self, input_size, hidden_shape, cd_k, stddev):
        self.input_size = input_size
        self.hidden_shape = hidden_shape
        self.cd_k = cd_k
        self.stddev = stddev
        self.layer_parameters = [{'W':None, 'bp':None, 'bn':None, 'stddev': None} for _ in range(len(hidden_shape))]

    def train(self, data, epochs, batch_size, lerning_rate,  momentum):
        for index in range(len(self.layer_parameters)):
            if index==0:
                vn = self.input_size
            else:
                vn = self.hidden_shape[index-1]
            hn = self.hidden_shape[index]
            
            # Dane wejściowe są ciągłe, więc obliczamy specjalnie pierwszą warstwę
            if index==0:
                gb_rbm = GB_RBM(vn, hn, self.cd_k, self.stddev)
                gb_rbm.train(data, epochs, batch_size, lerning_rate, momentum)

                # Przepisujemy zoptymalizowane wartości parametrów z modelu do tablicy
                self.layer_parameters[index]['W'] = gb_rbm.weights.clone()
                self.layer_parameters[index]['bp'] = gb_rbm.bias_p.clone()
                self.layer_parameters[index]['bn'] = gb_rbm.bias_n.clone()
                self.layer_parameters[index]['stddev'] = gb_rbm.v_stddev.clone()
            else:
                rbm = RBM(vn, hn, self.cd_k)
                # Generujemy wartości na warstwie poprzedniej
                data_dash = self.generate_input_for_layer(index,data)
                rbm.train(data_dash, epochs, batch_size, lerning_rate, momentum)
                self.layer_parameters[index]['W'] = rbm.weights.clone()
                self.layer_parameters[index]['bp'] = rbm.bias_p.clone()
                self.layer_parameters[index]['bn'] = rbm.bias_n.clone()

            print("Finished Training Layer:", index, "to", index+1)


    # Oblicza wartości na ukrytej warstwie numer "layer", mając dane z poprzedniej jako input 
    def forward_pass(self, layer, input):
        if layer == 0:
            activation = (input @ self.layer_parameters[layer]['W']) / (self.layer_parameters[layer]['stddev']**2) + self.layer_parameters[layer]['bp']
        else:
            activation = (input @ self.layer_parameters[layer]['W']) + self.layer_parameters[layer]['bp']
        probablity = torch.sigmoid(activation)
        h = torch.bernoulli(probablity)
        return probablity, h
    
    def backward_pass(self, layer, hidden):
        activation = hidden @ self.layer_parameters[layer]['W'].T + self.layer_parameters[layer]['bn']
        if layer == 0:
            v = torch.normal(mean=activation, std=self.layer_parameters[layer]['stddev'])
        else:
            probablity = torch.sigmoid(activation)
            v = torch.bernoulli(probablity)
        return v


    def evaluate(self, dataset):
        error = torch.zeros(dataset.size(1))
        for idx in range(dataset.size(0)):
            error += (dataset[idx, :] - self.reconstruct(dataset[idx, :]))**2
        mean_error = torch.mean(error)/dataset.size(0)
        print(f"Evaluation error: {mean_error}")
        return mean_error
    
    def generate_input_for_layer(self, index, data):
        x_dash = data.clone()
        for i in range(index):
               _, x_dash = self.forward_pass(i, x_dash)
        return x_dash
        
    def reconstruct(self, observation):
        observation = torch.as_tensor(observation, dtype=torch.double)
        visible = self.generate_input_for_layer(len(self.layer_parameters), observation)
        layers = len(self.layer_parameters)
        layers = int(layers)
        for index in range(0, layers):
            visible = self.backward_pass(layers - 1 - index, visible)

        return visible
            