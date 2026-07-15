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
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rates, use_batch_norm=False):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        self.model = self._mlp()

    def forward(self, x):
        return self.model(x)

    def _mlp(self):
        layers = []
        prev_dim = self.input_dim
        for i in range(len(self.hidden_units)):
            layers.append(nn.Linear(prev_dim, self.hidden_units[i]))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_units[i]))
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


def ccc_loss(pred, target):
    """Concordance Correlation Coefficient loss: 1 - CCC."""
    pred_mean = pred.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    cov = (pred_centered * target_centered).mean(dim=0)
    var_pred = (pred_centered ** 2).mean(dim=0)
    var_target = (target_centered ** 2).mean(dim=0)
    mean_diff = (pred_mean - target_mean).squeeze(0) ** 2

    ccc = (2 * cov) / (var_pred + var_target + mean_diff + 1e-8)
    return (1 - ccc).mean()


def combined_loss(pred, target, alpha=0.5):
    """Combined MSE + CCC loss."""
    return alpha * F.mse_loss(pred, target) + (1 - alpha) * ccc_loss(pred, target)


def cross_entropy_loss(pred, target, eps=1e-8):
    """Cross-entropy loss for probability targets."""
    return -(target * torch.log(pred + eps)).sum(dim=1).mean()


LOSS_REGISTRY = {
    "l1": F.l1_loss,
    "mse": F.mse_loss,
    "ccc": ccc_loss,
    "combined": lambda p, t: combined_loss(p, t, alpha=0.5),
    "cross_entropy": cross_entropy_loss,
}


class scaden():
    @classmethod
    def from_file(cls, load_path):
        obj = cls.__new__(cls)
        obj.load_model(load_path)
        return obj

    def __init__(self, architectures, dropouts, train_x, train_y, lr=1e-4, batch_size=128, epochs=20, loss_fn="l1", weight_decay=0.0, patience=None, val_x=None, val_y=None, use_batch_norm=False, lr_scheduler="none", noise_std=0.0):
        self.architectures = architectures      # list of list[int], one per model
        self.dropouts = dropouts                # list of list[float], one per model
        self.models = []
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.inputdim = train_x.shape[1]
        self.outputdim = train_y.shape[1]
        self.train_loader = DataLoader(simdatset(train_x, train_y), batch_size=batch_size, shuffle=True)
        if val_x is not None and val_y is not None:
            self.val_loader = DataLoader(simdatset(val_x, val_y), batch_size=batch_size, shuffle=False)
        else:
            self.val_loader = None
        self.gene_names = None
        self.label_names = None
        if loss_fn not in LOSS_REGISTRY:
            raise ValueError(f"Unknown loss_fn '{loss_fn}'. Choose from {list(LOSS_REGISTRY.keys())}")
        self.loss_fn = loss_fn
        self.use_batch_norm = use_batch_norm
        self.lr_scheduler = lr_scheduler
        self.noise_std = noise_std

    def _subtrain(self, model, optimizer):
        model.train()
        epoch_losses = []
        best_val_loss = float('inf')
        best_state_dict = None
        epochs_no_improve = 0
        early_stopped = False
        best_epoch = 0

        scheduler = None
        if self.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
            )
        elif self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=1e-7
            )

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            epoch_loss = 0.0
            n_batches = 0
            for data, label in self.train_loader:
                if self.noise_std > 0:
                    data = data + torch.randn_like(data) * self.noise_std
                optimizer.zero_grad()
                batch_loss = self._compute_loss(model(data), label)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                n_batches += 1
            epoch_loss /= n_batches
            epoch_losses.append(epoch_loss)
            pbar.set_description(f'Epoch {epoch+1}/{self.epochs}, loss={epoch_loss:.4f}')

            # Early stopping check after each epoch
            if self.patience is not None and self.val_loader is not None:
                model.eval()
                val_loss = 0.0
                n_val_batches = 0
                with torch.no_grad():
                    for val_data, val_label in self.val_loader:
                        val_loss += self._compute_loss(model(val_data), val_label).item()
                        n_val_batches += 1
                val_loss /= n_val_batches
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                    best_epoch = epoch
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        early_stopped = True
                        model.load_state_dict(best_state_dict)
                        model = model.to(device)
                        pbar.set_description(f'Epoch {epoch+1}/{self.epochs}, loss={epoch_loss:.4f} (early stopped at epoch {best_epoch+1})')
                        break

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if self.patience is not None and self.val_loader is not None:
                        scheduler.step(val_loss)
                    else:
                        scheduler.step(epoch_loss)
                else:
                    scheduler.step()

        return model, epoch_losses, early_stopped, best_epoch

    def _compute_loss(self, pred, target):
        """Compute loss using the configured loss function."""
        return LOSS_REGISTRY[self.loss_fn](pred, target)

    def train(self):
        self.build_models()
        for i, model in enumerate(self.models):
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, eps=1e-07, weight_decay=self.weight_decay)
            print(f'Training model {i+1}/{len(self.models)} ...')
            self.models[i], epoch_losses, early_stopped, best_epoch = self._subtrain(model, optimizer)
            if early_stopped:
                print(f'  Model {i+1}: early stopped at epoch {best_epoch+1}, best loss={epoch_losses[best_epoch]:.4f}')
            else:
                print(f'  Model {i+1}: final loss={epoch_losses[-1]:.4f}')
        print('Training is done')

    def build_models(self):
        self.models = []
        for hidden_units, dropout_rates in zip(self.architectures, self.dropouts):
            model = MLP(self.inputdim, self.outputdim, hidden_units, dropout_rates, use_batch_norm=self.use_batch_norm)
            model = model.to(device)
            model.apply(initialize_weight)
            self.models.append(model)

    def predict(self, test_x):
        test_x = torch.from_numpy(test_x).to(device).float()
        pred_sum = None
        for model in self.models:
            model.eval()
            pred = model(test_x)
            if pred_sum is None:
                pred_sum = pred
            else:
                pred_sum += pred
        return (pred_sum / len(self.models)).cpu().detach().numpy()

    def save_model(self, path, genes_names: list, label_names: list):
        torch.save({
            "inputdim": self.inputdim,
            "outputdim": self.outputdim,
            "architectures": self.architectures,
            "dropouts": self.dropouts,
            "loss_fn": self.loss_fn,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "use_batch_norm": self.use_batch_norm,
            "lr_scheduler": self.lr_scheduler,
            "noise_std": self.noise_std,
            "genes_names": genes_names,
            "label_names": label_names
        }, path + '/architecture.pt')
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path + f'/model_{i}.pt')

    def load_model(self, path):
        arch = torch.load(path + '/architecture.pt', map_location='cpu')
        self.inputdim = arch['inputdim']
        self.outputdim = arch['outputdim']
        self.architectures = arch['architectures']
        self.dropouts = arch['dropouts']
        self.loss_fn = arch.get('loss_fn', 'l1')
        self.weight_decay = arch.get('weight_decay', 0.0)
        self.patience = arch.get('patience', None)
        self.use_batch_norm = arch.get('use_batch_norm', False)
        self.lr_scheduler = arch.get('lr_scheduler', 'none')
        self.noise_std = arch.get('noise_std', 0.0)
        self.val_loader = None
        self.gene_names = arch['genes_names']
        self.label_names = arch['label_names']
        self.build_models()
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(path + f'/model_{i}.pt', map_location='cpu'))


def reproducibility(seed=9):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
