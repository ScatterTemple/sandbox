import warnings
import pickle


import numpy as np
import optuna
from optuna.integration import BoTorchSampler

import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

import gpytorch
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

import plotly.graph_objects as go


warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)


def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    return [(x - 2) ** 2 + (y + 5) ** 2, x**2]


def optimize():
    sampler = BoTorchSampler(seed=42)

    study = optuna.create_study(sampler=sampler, directions=['minimize']*2)
    study.optimize(objective, n_trials=15)

    with open('study.pkl', 'wb') as f:
        pickle.dump(study, f)


class MyStandardScaler:

    # noinspection PyAttributeOutsideInit
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.m = x.numpy().mean(axis=0)
        # self.s = np.array([col.std() for col in x.numpy().T]).T
        self.s = x.numpy().std(axis=0, ddof=1)
        return torch.tensor(self.transform(x))

    def transform(self, x: np.ndarray or torch.Tensor) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return (x - self.m) / self.s
        else:
            return (x.numpy() - self.m) / self.s

    def inverse_transform(self, x: np.ndarray or torch.Tensor) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x * self.s + self.m
        else:
            return x.numpy() * self.s + self.m


class MyMinMaxScaler:

    # noinspection PyAttributeOutsideInit
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.max = x.numpy().max(axis=0)
        self.min = x.numpy().min(axis=0)
        return torch.tensor(self.transform(x))

    def transform(self, x: torch.Tensor) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return (x - self.min) / (self.max - self.min)
        else:
            return (x.numpy() - self.min) / (self.max - self.min)

    def inverse_transform(self, x: torch.Tensor) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x * (self.max - self.min) + self.min
        else:
            return x.numpy() * (self.max - self.min) + self.min


def train(study):

    # Extract the training data
    train_x = torch.tensor([[trial.params['x'], trial.params['y']] for trial in study.trials]).double()
    train_y = torch.tensor([trial.values for trial in study.trials]).double()  # .unsqueeze(-1)

    # Normalize the input data to the unit cube
    # scaler_x = MinMaxScaler()
    # train_x = torch.tensor(scaler_x.fit_transform(train_x)).double()
    scaler_x = MyMinMaxScaler()
    train_x = scaler_x.fit_transform(train_x)

    # Standardize the output data
    # scaler_y = StandardScaler()
    # train_y = torch.tensor(scaler_y.fit_transform(train_y)).double()
    scaler_y = MyStandardScaler()
    train_y = scaler_y.fit_transform(train_y)

    # Fit a Gaussian Process model using the extracted data
    gp = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # save model
    with open('fit-model.pkl', 'wb') as f:
        pickle.dump([gp, scaler_x, scaler_y], f)


if __name__ == '__main__':
    # # run optimize
    # optimize()
    #
    # # train model
    # with open('study.pkl', 'rb') as f:
    #     study = pickle.load(f)
    # train(study)

    # load model
    with open('fit-model.pkl', 'rb') as f:
        gp, scaler_x, scaler_y = pickle.load(f)

    # predict something
    test_x = torch.tensor([[2., -5.], [0., 0.], [10, 10]]).double()
    gp.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.tensor(scaler_x.transform(test_x)).double()
        pred = gp(test_x)
        # mean = scaler_y.inverse_transform(pred.mean.numpy().reshape(1, -1))
        # variance = pred.variance.numpy() * scaler_y.scale_ ** 2
        mean = scaler_y.inverse_transform(torch.permute(pred.mean, (1, 0)))
        variance = torch.permute(pred.variance, (1, 0)).numpy() * scaler_y.s ** 2
        for x, m, v in zip(scaler_x.inverse_transform(test_x), mean, variance):
            print(f'Y of {x} is estimated {m}±{np.sqrt(v)}')

    # ===== show =====
    # Create a grid of input values
    x_grid = np.linspace(-10, 10, 100)
    y_grid = np.linspace(-10, 10, 100)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()])
    # Normalize the grid input data
    grid_normalized = torch.tensor(scaler_x.transform(grid)).double()
    # Predict using the model
    gp.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = gp(grid_normalized)
        mean = scaler_y.inverse_transform(torch.permute(pred.mean, (1, 0)))
        std_dev = np.sqrt(torch.permute(pred.variance, (1, 0)).numpy()) * scaler_y.s

    # Reshape the mean to match the grid shape
    zz = mean[:, 0].reshape(xx.shape)
    zz_std_dev = std_dev[:, 0].reshape(xx.shape)
    zz_upper = zz + zz_std_dev
    zz_lower = zz - zz_std_dev

    # Create the surface plot
    fig = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy)])
    contours = dict(x=dict(highlight=False, show=True, color='blue',
                           start=-3, end=3, size=0.3),
                    y=dict(highlight=False, show=True, color='blue',
                           start=-3, end=3, size=0.3),
                    z=dict(highlight=False, show=False))
    fig.add_trace(go.Surface(z=zz_upper, x=xx, y=yy, showscale=False, opacity=0.5, contours=contours))
    fig.add_trace(go.Surface(z=zz_lower, x=xx, y=yy, showscale=False, opacity=0.5, contours=contours))

    fig.update_layout(title='Objective Function Surface',
                      scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='Objective'))

    fig.show()
