import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

input_size = 784  # MNIST input is 784
hidden_size = 400  # Estimate from `(input_size + output_size) % 2`
out_size = 10  # 0 - 9 they are 10 classes
epochs = 10  #
batch_size = 100
learning_rate = 0.001
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):  # Forward Propagate
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
