from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.special import expit as sigmoid
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import BinaryAUROC


np.random.seed(0)
torch.manual_seed(0)


def gen_samples(num_samples: int, num_features: int) -> tuple[torch.Tensor, ...]:
    X, y = make_classification(n_samples=num_samples, n_features=num_features, 
                               n_classes=2, weights=[0.8, 0.2])
    X = torch.from_numpy(X).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    y = y[:, None]
    # print(f'type of X: {X.dtype}[{X.shape}], type of y: {y.dtype}[{y.shape}], positive samples: {y.mean()}')
    return X, y
    X = torch.normal(0, 1, size=(num_samples, num_features))
    w = torch.normal(0, 1, size=(num_features, 1))
    bias = torch.normal(0, 1, size=(1,))
    z = X + torch.normal(0, 0.1, size=X.shape)
    # z = X
    z = X @ w + bias
    z = sigmoid(z)
    y = torch.where(z > 0.7, 1., 0.)
    # print(f'type of X: {X.dtype}[{X.shape}], type of y: {y.dtype}[{y.shape}]')

    return X, y


class Net(nn.Module):
    
    def __init__(self, input_dim: int, zero_weights: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        if zero_weights:
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
          eval_dataset: DataLoader):
    for epoch in range(10):
        running_loss = 0.
        train_auc = BinaryAUROC()
        for i, (inputs, labels) in enumerate(dataset):
            model.train()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_auc.update(outputs, labels)
            
            if i % 50 == 0:
                rate = evaluate(eval_dataset, model, optimizer, criterion)
                print(f'{epoch = }/{i:02d}, train auc: {train_auc.compute()}, train loss: {loss.item():.3f}, '
                      f'evaluate rate: {rate}')
                


def evaluate(dataset: DataLoader, model: Module, optimizer, criterion):
    model.eval()
    total, correct = 0, 0
    running_loss = 0.
    eval_auc = BinaryAUROC()
    with torch.no_grad():
        for inputs, labels in dataset:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            running_loss += loss.item() * labels.size(0)
            eval_auc.update(outputs, labels)
    eval_auc_score = eval_auc.compute().item()
    return eval_auc_score, running_loss / total


def main():
        
    num_samples = 15000
    num_features = 30
    X, y = gen_samples(num_features=num_features, num_samples=num_samples)
    
    dataset = MyDataset(X, y)
    train_dataset, valid_dataset = random_split(dataset, [2./3, 1./3])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    net = Net(input_dim=num_features, zero_weights=True)
    print(net)
    print(net.linear.weight)
    print(net.linear.bias)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, )
    train(train_loader, net, optimizer, criterion, eval_dataset=valid_loader)
    print(net)
    print(net.linear.weight)
    print(net.linear.bias)


if __name__ == '__main__':
    main()