import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    # Lecture example
    transforms.Normalize((0.,),(1.,))
])

# Download and load the training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Example: get a batch of training data
images, labels = next(iter(train_loader))
print("Batch of images shape:", images.shape)
# What does the image represent?
print("Batch of labels shape:", labels.shape)

"""
Commented out for just SimpleNN example testing.
print(labels[13])
plt.imshow(images[13].squeeze(), cmap='gray')
plt.show()
"""

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.l1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        # -1: figure it out for me.
        # Note both nn.Linear() and reshape() is okay with implied batch size!!
        x = x.reshape(-1, 28*28)
        output1 = self.l1(x)
        input2 = self.relu(output1)
        output2 = self.l2(input2)

        return output2


mynetwork = SimpleNN()
# Attempt: Call a layer 1 on a batch of data and print output shape
# Output shape (due to layer 2 call specified to 10 features) will be in blocks of 10?
print(mynetwork.forward(images))

# Batch size of 64 with 10 output parameters
print(mynetwork(images).shape)

# If model is preferable, keep the parameters for future testing.
params = list(mynetwork.parameters())
print(params)

# 1st layer (params[0]) - matrix
# 2nd layer (params[1]) - bias
# 3rd layer (params[2]) - ?
print(params[0].shape)

# Calculate number of parameters used in the model
num_params = sum(p.numel() for p in mynetwork.parameters())
print(f'Number of parameters: {num_params}')