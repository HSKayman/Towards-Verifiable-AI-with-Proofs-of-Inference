# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import pandas as pd


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
data = load_iris()

# %%
# Generate synthetic dataset
# X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
#                          n_redundant=5, n_classes=3, random_state=42)

X = data.data
y = data.target

# %%
X,y

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
len(X_train)

# %%
class ANN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], num_classes)
        self.relu = nn.ReLU()    
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %%
verified_model = ANN(input_size=4, hidden_sizes=[64, 32], num_classes=3).to(device)

# %%
# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(verified_model.parameters(), lr=0.001)

# %%
# Training loop
epochs = 30
for epoch in range(epochs):
    verified_model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = verified_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}')

# %%
# Evaluation
verified_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = verified_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

verified_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = verified_model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Train Accuracy: {100 * correct / total:.2f}%')


# %%
activations = {}
def get_activation(name, storage_dict):
    def hook(model, input, output):
        storage_dict[name] = output.detach()
    return hook

verified_model.fc1.register_forward_hook(get_activation('fc1', activations))
verified_model.fc2.register_forward_hook(get_activation('fc2', activations))
verified_model.fc3.register_forward_hook(get_activation('fc3', activations))

# %%
calibration_data = X_train[0]

# Get verified activations from original model
verified_model.eval()
with torch.no_grad():
    _ = verified_model(calibration_data)
    target_activations = {k: v.clone() for k, v in activations.items()}

real_activations = target_activations.copy()

# %% [markdown]
# # ADVERSARY SETUP

# %%
def crack_input(target_output, model, learning_rate=0.001, iterations=10000):
    # Initialize random input (3 features)
    input_tensor = torch.rand(1, 4, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)
    
    losses = []
    for i in range(iterations): 
        optimizer.zero_grad()
    
        # Forward pass
        predicted_output = model(input_tensor)
        
        # Compute loss
        loss = F.mse_loss(predicted_output, target_output)
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Update inputs
        optimizer.step()
        
        # Optional: print progress if loss zero hack is successful!!
        if i % 1000 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.6f}")

        if loss.item() < 1e-6:
            print("Loss is very low, stoped early.")
            break
    return input_tensor.detach()

# %%
#testing
pred_inputs = crack_input(real_activations['fc3'], verified_model, learning_rate=0.001, iterations=1)

# %% [markdown]
# # MULTIPLE INPUT with N ROUND

# %%
ROUND = 50
N_INPUTS = 120
results = pd.DataFrame(columns=[
    'input_id', 'round_id', 
    'fc1_min_abs_diff', 'fc1_max_abs_diff', 'fc1_mean_abs_diff',
    'fc2_min_abs_diff', 'fc2_max_abs_diff', 'fc2_mean_abs_diff',
    'fc3_min_abs_diff', 'fc3_max_abs_diff', 'fc3_mean_abs_diff',
    'real_input', "pred_input"
])
for i in range(N_INPUTS):
    print(f"Input {i+1}")
    for j in range(ROUND):
        print(f"Round {j+1}")

        # Registering hooks to capture activations
        activations = {}
        verified_model.fc1.register_forward_hook(get_activation('fc1', activations))
        verified_model.fc2.register_forward_hook(get_activation('fc2', activations))
        verified_model.fc3.register_forward_hook(get_activation('fc3', activations))

        calibration_data = X_train[i]

        # Get verified activations from original model
        verified_model.eval()
        with torch.no_grad():
            _ = verified_model(calibration_data)
            target_activations = {k: v.clone() for k, v in activations.items()}

        real_activations = target_activations.copy()


        # ADVERSARIAL INPUT GENERATION
        pred_inputs = crack_input(real_activations['fc3'], verified_model, learning_rate=0.005, iterations=10000)

        # Registering hooks to capture activations
        activations = {}
        verified_model.fc1.register_forward_hook(get_activation('fc1', activations))
        verified_model.fc2.register_forward_hook(get_activation('fc2', activations))
        verified_model.fc3.register_forward_hook(get_activation('fc3', activations))

        calibration_data = pred_inputs

        # ACTIVATIONS that ADVERSARIAL INPUT GENERATED
        verified_model.eval()
        with torch.no_grad():
            _ = verified_model(calibration_data)
            pred_activations = {k: v.clone() for k, v in activations.items()}

        round_results = {'input_id': i+1, 'round_id': j+1}
        
        # Compare and visualize the activations using absolute difference
        for layer in real_activations.keys():

            # Calculate mean absolute error between real and predicted activations
            abs_diff = torch.abs(real_activations[layer] - pred_activations[layer])
            mean_abs_diff = abs_diff.mean().item()
            max_abs_diff = abs_diff.max().item()
            min_abs_diff = abs_diff.min().item()

            # Store in results dictionary
            round_results[f'{layer}_min_abs_diff'] = min_abs_diff
            round_results[f'{layer}_max_abs_diff'] = max_abs_diff
            round_results[f'{layer}_mean_abs_diff'] = mean_abs_diff

        # Append results to DataFrame
        round_results['real_input'] = X_train[i].cpu().flatten().numpy().tolist()
        round_results['pred_input'] = pred_inputs.cpu().flatten().numpy().tolist()
        results = pd.concat([results, pd.DataFrame([round_results])], ignore_index=True)
        
            
results.to_csv('activation_diff_results2.csv', index=False)


