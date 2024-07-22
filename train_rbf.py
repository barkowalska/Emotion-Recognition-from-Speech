import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data import preprocess_emodb_data
from RBF import RBF

emodb_directory = 'C:\\Users\\jkowa\\Desktop\\glosowe\\data1'

X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = preprocess_emodb_data(emodb_directory)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

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
    'num_centers': [50, 75, 100, 125, 150, 175, 200],
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

X_full_train = np.vstack((X_train, X_val))
y_full_train = np.hstack((y_train, y_val))
input_dim = X_full_train.shape[1]
output_dim = len(np.unique(y_full_train))
rbf = RBF(input_dim, best_params['num_centers'], output_dim)
rbf.beta = best_params['beta']

Y_full_train = np.eye(output_dim)[y_full_train]
rbf.train(X_full_train, Y_full_train)

y_test_pred = rbf.predict(X_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

test_accuracy = accuracy_score(y_test, y_test_pred_labels)
print(f"Test Accuracy with best model: {test_accuracy * 100:.2f}%")
