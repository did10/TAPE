import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


class simdatset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)
        y = torch.from_numpy(self.Y[index]).float().to(device)
        return x, y


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rates):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self._mlp()

    def forward(self, x):
        return self.model(x)

    def _mlp(self):
        layers = []
        prev_dim = self.input_dim
        for i in range(len(self.hidden_units)):
            layers.append(nn.Linear(prev_dim, self.hidden_units[i]))
            layers.append(nn.Dropout(self.dropout_rates[i]))
            layers.append(nn.ReLU())
            prev_dim = self.hidden_units[i]
        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)


def initialize_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class scaden():
    @classmethod
    def from_file(cls, load_path):
        obj = cls.__new__(cls)
        obj.load_model(load_path)
        return obj

    def __init__(self, hidden_units, dropout_rates, train_x, train_y, lr=1e-4, batch_size=128, epochs=20):
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.model = None
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.inputdim = train_x.shape[1]
        self.outputdim = train_y.shape[1]
        self.train_loader = DataLoader(simdatset(train_x, train_y), batch_size=batch_size, shuffle=True)
        self.gene_names = None
        self.label_names = None

    def _subtrain(self, model, optimizer):
        model.train()
        loss = []
        for _ in tqdm(range(self.epochs)):
            for data, label in self.train_loader:
                optimizer.zero_grad()
                batch_loss = F.l1_loss(model(data), label)
                batch_loss.backward()
                optimizer.step()
                loss.append(batch_loss.cpu().detach().numpy())
        return model, loss

    def train(self):
        self.build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-07)
        print('Training model...')
        self.model, loss = self._subtrain(self.model, optimizer)
        print('Training is done')

    def build_model(self):
        self.model = MLP(self.inputdim, self.outputdim, self.hidden_units, self.dropout_rates)
        self.model = self.model.to(device)
        self.model.apply(initialize_weight)

    def predict(self, test_x):
        test_x = torch.from_numpy(test_x).to(device).float()
        self.model.eval()
        pred = self.model(test_x)
        return pred.cpu().detach().numpy()

    def save_model(self, path, genes_names: list, label_names: list):
        torch.save({
            "inputdim": self.inputdim,
            "outputdim": self.outputdim,
            "hidden_units": self.hidden_units,
            "dropout_rates": self.dropout_rates,
            "genes_names": genes_names,
            "label_names": label_names
        }, path + '/architecture.pt')
        torch.save(self.model.state_dict(), path + '/model.pt')

    def load_model(self, path):
        architecture = torch.load(path + '/architecture.pt', map_location='cpu')
        self.inputdim = architecture['inputdim']
        self.outputdim = architecture['outputdim']
        self.hidden_units = architecture['hidden_units']
        self.dropout_rates = architecture['dropout_rates']
        self.gene_names = architecture['genes_names']
        self.label_names = architecture['label_names']
        self.build_model()
        self.model.load_state_dict(torch.load(path + '/model.pt', map_location='cpu'))


def reproducibility(seed=9):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
