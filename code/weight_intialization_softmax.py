from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.special import expit as sigmoid
from torch.utils.data import DataLoader, Dataset, random_split


np.random.seed(0)
torch.manual_seed(0)


def gen_samples(num_samples: int, num_features: int, num_classes: int) -> tuple[torch.Tensor, ...]:
    X, y = make_classification(n_samples=num_samples, n_features=num_features, n_informative=4,
                               n_classes=num_classes)
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.long)
    return X, y


class Net(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, zero_weights: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        if zero_weights:
            # zero out weights and bias
            self.linear.weight.data.zero_()
            self.linear.bias.data.zero_()
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class MyDataset(Dataset):
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
        assert len(X) == len(y), 'mismatched number of samples and labels.'
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train(dataset: DataLoader, model: Module, optimizer, criterion, 
          eval_dataset: DataLoader, num_epoches: int = 1):
    for epoch in range(num_epoches):
        running_loss = 0.
        for i, (inputs, labels) in enumerate(dataset):
            model.train()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 50 == 0:
                rate = evaluate(eval_dataset, model, optimizer, criterion)
                print(f'{epoch = }/{i:02d}, train loss: {loss.item():.3f}, '
                      f'evaluate rate: {rate}')


def evaluate(dataset: DataLoader, model: Module, optimizer, criterion):
    model.eval()
    total, correct = 0, 0
    running_loss = 0.
    with torch.no_grad():
        for inputs, labels in dataset:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)
    return 0., running_loss / total


def main():
        
    num_samples = 15000
    num_features = 30
    num_classes = 5
    X, y = gen_samples(num_features=num_features, num_samples=num_samples, num_classes=num_classes)
    
    dataset = MyDataset(X, y)
    train_dataset, valid_dataset = random_split(dataset, [2./3, 1./3])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    net = Net(input_dim=num_features, output_dim=num_classes, zero_weights=True)
    print('before training')
    print(net)
    print(net.linear.weight)
    print(net.linear.bias)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, )
    train(train_loader, net, optimizer, criterion, eval_dataset=valid_loader, num_epoches=20)
    print('after training')
    print(net)
    print(net.linear.weight)
    print(net.linear.bias)


if __name__ == '__main__':
    main()