import warnings

import numpy as np

import optuna
from optuna.integration.botorch import BoTorchSampler

import torch
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import gen_candidates_scipy

from helper import NonOverwritablePartial

import optuna_integration.botorch


warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def get_constraint_dict():

    inequality_constraints: "Optional[List[Tuple[Tensor, Tensor, float]]]" = None
    equality_constraints: "Optional[List[Tuple[Tensor, Tensor, float]]]" = None
    nonlinear_inequality_constraints: "Optional[List[Tuple[Callable, bool]]]" = None

    nonlinear_inequality_constraints = [
        (
            lambda x: ineq_cns(
                *x
            ),  # `callable(x) >= 0` という形式の制約を表す呼び出し可能オブジェクト。点内制約の場合、`callable()` は形状 `d` の 1 次元テンソルを受け取り、スカラーを返します。点間制約の場合、`callable()` は形状 `q x d` の 2 次元テンソルを受け取り、やはりスカラーを返します。
            True,  # 点内制約か点間制約かを示します (点内制約の場合は `True`、点間制約の場合は `False`)。バッチ内の各候補に同じ制約を適用する場合 は点内制約。
        )
    ]

    equality_constraints = [
        (
            (
                indices := torch.tensor([0])  # r
            ),  # 入力 prm のうち何番目の変数を使うかのインデックスのリスト。長さ L とする。
            (
                coefficients := torch.tensor([1]).double()
            ),  # 入力 prm から indices で抽出された配列と内積を取る係数配列。長さ L でないとダメ。
            (rhs := 1.0),  # 内積がとるべき合計値。int。
        ),
    ]

    constraints_dict = dict(
        inequality_constraints=inequality_constraints,
        # equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        ic_generator=gen_batch_initial_conditions,
        gen_candidates=gen_candidates_scipy,
    )

    return constraints_dict


def objective(trial: optuna.Trial):
    r = trial.suggest_float("r", 0, 1)
    theta = trial.suggest_float("theta", 0, 2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def ineq_cns(r, theta: torch.Tensor):
    print()
    print(np.pi - theta)
    print(np.pi)
    print(theta)
    return np.pi - theta  # theta <= np.pi is feasible


def do_mock(constraints_dict):
    original_fun = optuna_integration.botorch.optimize_acqf
    overwritten_and_constrainted_fun = NonOverwritablePartial(
        original_fun,
        **dict(
            options={"batch_limit": 1, "maxiter": 200, "nonnegative": True},
            **constraints_dict,
        ),
    )
    optuna_integration.botorch.optimize_acqf = overwritten_and_constrainted_fun


if __name__ == "__main__":

    constraints_dict = get_constraint_dict()

    do_mock(constraints_dict)

    sampler = BoTorchSampler()

    study = optuna.create_study(sampler=sampler, directions=["minimize"] * 2)

    study.optimize(
        objective,
        n_trials=20,
    )
