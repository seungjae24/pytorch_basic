import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

dataset = MNIST(root='data/', download=True)
test_dataset = MNIST(root='data/', train=False)

#image, label = dataset[0]
#plt.imshow(image, cmap='gray')
#plt.show()
#print('Label: ', label)

dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

#img_tensor, label = dataset[0]
#print(img_tensor.shape, label)

#print(img_tensor[0,10:15,10:15])
#print(torch.max(img_tensor), torch.min(img_tensor))

#plt.imshow(img_tensor[0,10:15,10:15], cmap='gray')
#plt.show()

train_ds, val_ds = random_split(dataset, [50000, 10000])

# 50000, 10000 number of train datasets & validation datasets
#print(len(train_ds), len(val_ds))

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

input_size = 28*28 # A image can be represented as a vector, which have a size of 784.
num_classes = 10  # number of target label is 0-9.

# Logistic regression
# model = nn.Linear(input_size, num_classes)

# # torch.Size([10, 784])
# print(model.weight.shape)
# print(model.weight)

# # torch.Size([10])
# print(model.bias.shape)
# print(model.bias)

# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#     def say_hello(self):
#         print("Hello my name is " + self.name + "!")

# bob = Person("Bob", 32)
# print(bob.name, bob.age)

# bob.say_hello()

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

model = MnistModel()

# 
# print('\nmodel.linear : \n', model.linear)
# print('\nweight, bias : \n', model.linear.weight.shape, model.linear.bias.shape)

for images, labels in train_loader:
    print(images.shape)
    outputs = model(images)
    break

# print('\noutputs.shape : \n', outputs.shape)
# print('\nSample outputs :\n', outputs[:2].data)

# # Apply softmax fuction for each output row
# probs = F.softmax(outputs, dim=1)

# print("\nSample probabilities:\n", probs[:2].data)
# print("\nSum: \n", torch.sum(probs[0]).item())

# max_probs, preds = torch.max(probs, dim=1)

# print("\nmax_probs: \n", max_probs)

# # Very different results beacuse of randomly instantiated weight and bias
# print("\nPreds: \n", preds)
# print("\nLabels: \n", labels)


# Let's Train weight and bias

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

print("\nAccuracy: \n", accuracy(outputs, labels))

loss_fn = F.cross_entropy

# loss for current batch of data
loss = loss_fn(outputs, labels)

print("\nLoss: \n", loss)

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    
    for epoch in range(epochs):
        
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = MnistModel()


result0 = evaluate(model, val_loader)

history1 = fit(5, 0.001, model, train_loader, val_loader)
history2 = fit(5, 0.001, model, train_loader, val_loader)
history3 = fit(5, 0.001, model, train_loader, val_loader)
history4 = fit(5, 0.001, model, train_loader, val_loader)

history = [result0] + history1 + history2 + history3 + history4
accuracies = [result['val_acc'] for result in history]

plt.show()

plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
plt.show()

test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')


print('Label:', label, ', Predicted:', predict_image(img, model))