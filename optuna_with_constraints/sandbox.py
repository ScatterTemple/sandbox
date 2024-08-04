import torch
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.utils.transforms import standardize
from botorch.optim import optimize_acqf

# Define the objective function
def objective_function(x):
    return (x ** 2).sum(dim=-1)

# Define the constraint function
def my_constraint_fun(x):
    return 1 - x.sum(dim=-1)

# Example data
train_x = torch.rand(10, 2)
train_y = objective_function(train_x).unsqueeze(-1)

# Standardize the targets
train_y_standardized = standardize(train_y)

# Fit a Gaussian Process model
gp = SingleTaskGP(train_x, train_y_standardized)
gp = gp.to(train_x)

# Define the acquisition function
ei = ExpectedImprovement(model=gp, best_f=train_y_standardized.max())

# Define bounds
bounds = torch.stack([torch.zeros(2), torch.ones(2)])

# Optimize the acquisition function
best_point, _ = optimize_acqf(
    acq_function=ei,
    bounds=bounds,
    q=1,
    nonlinear_inequality_constraints=[(my_constraint_fun, True)],
    return_best_only=True,
    sequential=True,
    num_restarts=20,
)

print("Best point found:", best_point)
