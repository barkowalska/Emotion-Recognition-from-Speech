import torch
import numpy as np
from DBN import DBN
from RBF import RBF 
from sklearn.model_selection import train_test_split
import librosa
from data import   load_labels, load_signals
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


directory = 'C:/Users/jkowa/Desktop/glosowe/data1'  
dataset_type = 'emodb'  
#frames=load_frames(directory)

def compute_features(dbn, data):
    features = []

   
    for row in data:
        row=torch.as_tensor(row)
        feature = dbn.generate_input_for_layer(len(dbn.layer_parameters), row)
        features.append(feature)
    return np.array(features)


X = []  



dbn = DBN(400, (10, 50, 100), 5, 1) 
dbn.load_model('dbn_model.pth')

X= load_signals(directory)

Y=load_labels(directory, dataset_type )
X=compute_features(dbn, X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)
print("X=  ",X.shape)
#print("Y=  ", Y.shape)
print("Loaded model parameters:")
for i, layer in enumerate(dbn.layer_parameters):
    print(f"Layer {i}: W: {layer['W'].shape}, bp: {layer['bp'].shape}, bn: {layer['bn'].shape}, stddev: {layer['stddev'].shape if layer['stddev'] is not None else 'None'}")

X = torch.as_tensor(X)
#X = dbn.generate_input_for_layer(len(dbn.layer_parameters), X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

input_dim = int(X_train.shape[1])
output_dim = int(len(np.unique(y_train)))
num_centers = output_dim

def train_evaluate_rbf(X_train, y_train, X_val, y_val, num_centers, beta):
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    rbf = RBF(input_dim, num_centers, output_dim)
    rbf.beta = beta
    
    Y_train = np.eye(output_dim)[y_train]
    rbf.train(X_train, Y_train)
    
    y_val_pred = rbf.predict(X_val)
    y_val_pred_labels = np.argmax(y_val_pred, axis=1)
    
    val_accuracy = accuracy_score(y_val, y_val_pred_labels)
    return val_accuracy

param_grid = {
  'num_centers': [8, 12, 16, 20, 24, 28, 32, 36],
    'beta': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
}
 

best_val_accuracy = 0
best_params = {}

results = []

for num_centers in param_grid['num_centers']:
    for beta in param_grid['beta']:
        val_accuracy = train_evaluate_rbf(X_train, y_train, X_val, y_val, num_centers, beta)
        print(f"Validation Accuracy for num_centers={num_centers}, beta={beta}: {val_accuracy * 100:.2f}%")
        
        results.append((num_centers, beta, val_accuracy))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params['num_centers'] = num_centers
            best_params['beta'] = beta

print(f"Best Parameters: {best_params}")
print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
a=train_evaluate_rbf(X_train, y_train, X_val, y_val, num_centers, beta=0.5)

print(f"Validation Accuracy of DBN with RBF: {a * 100:.2f}%")