from RBM import GB_RBM
import pandas as pd
import numpy as np
import os
import librosa
from DBN import DBN
import torch
from sklearn.model_selection import train_test_split
from data import load_signals




directory = "C:/Users/jkowa/Desktop/glosowe/data1"

signals=load_signals(directory)

X_train, X_test = train_test_split(signals, train_size=0.3, test_size=0.3, random_state=42)



model=DBN(X_train.shape[1], (100, 50, 10  ), 5, 1)
model.train(data=X_train, epochs=10, batch_size=128, learning_rate=0.01, momentum=0.1)
epochs=100
batch_size=128
learning_rate=0.001
print("Model parameters before saving:")
for i, layer in enumerate(model.layer_parameters):
    print(f"Layer {i}: W: {layer['W'].shape}, bp: {layer['bp'].shape}, bn: {layer['bn'].shape}, stddev: {layer['stddev'].shape if layer['stddev'] is not None else 'None'}")



current_directory = os.getcwd()
model_file_path = os.path.join(current_directory, "dbn_model.pth")
model.save_model(model_file_path)

loaded_model = DBN(X_train.shape[1], (10, 50, 100), 5, 1)
loaded_model.load_model(model_file_path)


print("Loaded model parameters:")
for i, layer in enumerate(loaded_model.layer_parameters):
    print(f"Layer {i}: W: {layer['W'].shape}, bp: {layer['bp'].shape}, bn: {layer['bn'].shape}, stddev: {layer['stddev'].shape if layer['stddev'] is not None else 'None'}")

test_frame = torch.tensor(X_test[0], dtype=torch.float32)


hidden_activations = loaded_model.generate_input_for_layer(len(loaded_model.layer_parameters), test_frame)

print("Hidden Layer Activations from loaded model: ", hidden_activations)

print("epochs: ", epochs, "batchs size: ", batch_size, "learning rate: ", learning_rate )
evaluation_error_before_saving = model.evaluate(torch.tensor(X_test, dtype=torch.float32))
print(f"Evaluation error before saving: {evaluation_error_before_saving}")

