import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def model(x):
    ## @ means matrix multiplication in pytorch and .t means transpose of tensor
    return x @ w.t() + b

def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

print('\n ------------Output--------- \n')

# 1. Understand Gradient Descent Method

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]] , dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
#print(inputs)
#print(targets)

w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
#print(w)
#print(b)

preds = model(inputs)
loss = mse(preds, targets)

#print("\nbefore loss: \n", loss)

for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

preds = model(inputs)
loss = mse(preds, targets)

#print("\nafter loss: \n", loss)

#print("\npreds: \n", preds)
#print("\ntargets: \n", targets)

# 2. Using implements

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')


inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Training Data Set
train_ds = TensorDataset(inputs, targets)

#print("\ntrain_ds:\n", train_ds[:])

# Training Data Loader
train_dl = DataLoader(train_ds, batch_size=5, shuffle=True)

for xb, yb in train_dl:
    print(xb)
    print(yb)
    break

model = nn.Linear(3,2)

#print("\nmodel.weight:\n", model.weight)
#print("\nmodel.bias:\n", model.bias)
#list(model.parameters())

preds = model(inputs)

print("\npreds: \n", preds)

loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)
print(loss)

# SGD = stochastic gradient descent
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(100, model, loss_fn, opt, train_dl)

preds = model(inputs)
print("\npreds: \n", preds)
print("\ntargets: \n", targets)