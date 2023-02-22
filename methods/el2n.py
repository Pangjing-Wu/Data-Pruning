from typing import List, Union

import numpy as np
import torch
import torch.nn as nn


class EL2N(object):

    def __init__(self, model:nn.Module, order:Union[int,str], n_classes:int) -> None:
        self.model     = model
        self.order     = order
        self.n_classes = n_classes
        self.device    = next(iter(model.parameters())).device

    def score(self, x:torch.Tensor, y:torch.Tensor) -> np.ndarray:
        """x: batch of fratures, [batch size, feature]. y: batch of labels, [batch size]."""
        assert x.size(0) == y.size(0)
        with torch.no_grad():
            x        = x.to(self.device)
            logistic = torch.softmax(self.model(x), dim=-1).cpu().numpy()
            y        = self.__onehot_map(y).numpy()
        assert logistic.shape == y.shape
        return np.linalg.norm((logistic-y), ord=self.order, axis=-1)

    def scores(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        score  = list()
        for x, y in dataloader:
            score.append(self.score(x, y))
        return np.concatenate(score, axis=-1)

    def __onehot_map(self, y: torch.Tensor) -> torch.TensorType:
        return torch.eye(self.n_classes)[y.numpy()]


class GraNd(object):

    def __init__(self, model:nn.Module, order:Union[int,str]) -> None:
        self.model = model
        self.order = order
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0)
        self.criterion = nn.CrossEntropyLoss()
        self.device    = next(iter(model.parameters())).device

    def score(self, x:torch.Tensor, y:torch.Tensor) -> np.ndarray:
        """x: batch of fratures, [batch size, feature]. y: batch of labels, [batch size]."""
        assert x.size(0) == y.size(0)
        x, y  = x.to(self.device), y.to(self.device)
        x     = self.model.fx(x).detach()
        grand = list()
        for x_, y_ in zip(x, y):
            self.optimizer.zero_grad()
            logistic = torch.softmax(self.model.fc(x_), dim=-1)
            loss = self.criterion(logistic, y_)
            loss.backward()
            grad = self.model.fc.weight.grad.flatten().cpu().numpy()
            grand.append(np.linalg.norm(grad, ord=self.order, axis=-1))
        self.optimizer.zero_grad()
        return np.array(grand)

    def scores(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        score  = list()
        for x, y in dataloader:
            score.append(self.score(x, y))
        return np.concatenate(score, axis=-1)