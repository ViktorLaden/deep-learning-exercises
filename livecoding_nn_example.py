from copy import deepcopy
import numpy as np
import torch
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from mlxtend.data import mnist_data
from sklearn.model_selection import train_test_split

# training hyperparams
MAX_EPOCHS = 100
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 0
DROPOUT_RATE = 0.1

# load MNIST data from mlxtend.data -- what to include?
X, y = mnist_data()

# mask = (y == 0) | (y == 1)
# X, y = X[mask], y[mask]

# Make dataset splits -- what should be considered?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

X_train, X_val, X_test = tensor(X_train), tensor(X_val), tensor(X_test)
y_train, y_val, y_test = tensor(y_train), tensor(y_val), tensor(y_test)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Define model

model = nn.Sequential(
    nn.Linear(in_features=28**2, out_features=512),
    nn.ReLU(),
    nn.Dropout(p=DROPOUT_RATE),
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Dropout(p=DROPOUT_RATE),
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Dropout(p=DROPOUT_RATE),
    nn.Linear(in_features=128, out_features=10),
)

# Optimizer, schedule, and device (sneak peek: regularization?)
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
model.to(device)

# Make training loop -- when to stop? What to save?


def evaluate(model, loader, device):
    model.eval()
    with torch.no_grad():
        true_labels = np.empty(0)
        est_labels = np.empty(0)
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).long()
            logits = model(x)
            yhat = logits.argmax(dim=-1).detach().cpu().numpy()
            y = y.cpu().numpy()
            est_labels = np.append(est_labels, yhat)
            true_labels = np.append(true_labels, y)
    return np.mean(est_labels == true_labels)


best_val_acc = 0
for epoch in range(MAX_EPOCHS):
    model.train()
    true_labels = np.empty(0)
    est_labels = np.empty(0)
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device).float(), y.to(device).long()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        yhat = logits.argmax(dim=-1).detach().cpu().numpy()
        y = y.cpu().numpy()
        est_labels = np.append(est_labels, yhat)
        true_labels = np.append(true_labels, y)
    val_acc = evaluate(model, val_loader, device)
    train_acc = np.mean(true_labels == est_labels)

    if val_acc > best_val_acc:
        best_state = deepcopy(model.state_dict())
        best_val_acc = val_acc
        print("Updated best!")

    epoch_report = f"Epoch {epoch}: train acc={train_acc:.3f},"
    epoch_report += f"val acc={val_acc:.3f}."
    print(epoch_report)


# Make test
model.load_state_dict(best_state)
test_acc = evaluate(model, test_loader, device)
print(f"Test accuracy: {test_acc:.3f}")
