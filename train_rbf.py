import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from rbf_layer import RBFLayer, euclidean_norm, rbf_gaussian
from main import parse_filename, emotion_mapping

# Directory containing audio files
directory = 'glosowe/wav'

# Load data
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = parse_filename(directory)

# Number of classes
num_classes = len(np.unique(y_train))


num_features = X_train.shape[1]

rbf = RBFLayer(in_features_dim=num_features,
               num_kernels=3,
               out_features_dim=num_classes,
               radial_function=rbf_gaussian,
               norm_function=euclidean_norm,
               normalization=True)

optimiser = torch.optim.Adam(rbf.parameters(), lr=1e-3)
batch_size = 32
trn_losses = []
val_losses = []

for epoch in range(512):
    indices = np.random.permutation(X_train.shape[0])
    batch_idx = 0
    epoch_trn_losses = []
    epoch_val_losses = []
    
    # Epoch training
    while batch_idx < X_train.shape[0]:
        idxs = indices[batch_idx:batch_idx + batch_size]
        x = torch.tensor(X_train[idxs], dtype=torch.float32)
        labels = torch.tensor(y_train[idxs], dtype=torch.float32).flatten()
        
        # Compute loss
        optimiser.zero_grad()
        y = rbf(x)
        loss = nn.CrossEntropyLoss()(y, labels.long())
        epoch_trn_losses.append(loss.item())
        loss.backward()
        optimiser.step()
        batch_idx += batch_size
        
    trn_loss = np.mean(epoch_trn_losses)
    trn_losses.append(trn_loss)
    
    with torch.no_grad():
        # Compute validation
        x = torch.tensor(X_val, dtype=torch.float32)
        labels = torch.tensor(y_val, dtype=torch.float32).flatten()
        y = rbf(x)
        loss = nn.CrossEntropyLoss()(y, labels.long())
        epoch_val_losses.append(loss.item())
        
    val_loss = np.mean(epoch_val_losses)
    val_losses.append(val_loss)
    
    # Print results for each epoch
    print(f'Epoch [{epoch+1}/512], Training Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}')

# Final evaluation on test set
with torch.no_grad():
    x = torch.tensor(X_test, dtype=torch.float32)
    labels = torch.tensor(y_test, dtype=torch.float32).flatten()
    y = rbf(x)
    test_loss = nn.CrossEntropyLoss()(y, labels.long())
    print(f'Test Loss: {test_loss:.4f}')
