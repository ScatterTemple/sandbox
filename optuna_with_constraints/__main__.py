import warnings

import botorch.exceptions.warnings
import numpy as np

import optuna
from optuna.integration.botorch import BoTorchSampler

import torch
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.optimize import (
    gen_candidates_scipy,
    _optimize_acqf_sequential_q,
    OptimizeAcqfInputs,
    gen_one_shot_hvkg_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from botorch.utils.transforms import unnormalize, normalize

from helper import NonOverwritablePartial

import optuna_integration.botorch


warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.NumericsWarning)


def get_constraint_dict():

    inequality_constraints: "Optional[List[Tuple[Tensor, Tensor, float]]]" = None
    equality_constraints: "Optional[List[Tuple[Tensor, Tensor, float]]]" = None
    nonlinear_inequality_constraints: "Optional[List[Tuple[Callable, bool]]]" = None

    nonlinear_inequality_constraints = [
        (
            lambda x: ineq_cns(
                *(
                    unnormalize(x, bounds=torch.tensor([[0, 0], [1, 2 * np.pi]]))
                ).double()  # 多分ここで restore する必要がある
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

    def my_generator(
        *args,
        nonliner_constraints=None,
        **kwargs,
    ):
        # print("")
        # print("===== 初期値の評価開始 =====")

        counter = 0
        batch_initial_conditions = gen_batch_initial_conditions(*args, **kwargs)
        n_req = len(batch_initial_conditions)
        batch_initial_conditions_filterd = []
        while True:
            counter += 1
            if counter > 1000:
                # たった二変数一拘束ですらここを突破できない。
                # 初期値探索も多分完全なランダムではない。
                # 線形拘束なら動くのは、デタミニスティックに candidate できるからだろう。
                if len(batch_initial_conditions_filterd) == 0:
                    print(
                        'raise RuntimeError("拘束を満たす初期値が見つかりませんでした。")'
                    )
                    batch_initial_conditions_filterd = torch.tensor(
                        [[[0.5, 0.75]]]
                    )  # 本当はここ、前に成功したところにしたくない？ detaministic なので...
                    # from optuna.exceptions import TrialPruned
                    # raise TrialPruned
                else:
                    break

            for ic_candidate in batch_initial_conditions:
                # print("----- 初期値のチェック -----")
                # print(ic_candidate)
                # 拘束を満たさないものは飛ばす
                for cns in nonliner_constraints:
                    if cns[0](*ic_candidate) < 0:
                        # print("拘束が負なのでこの ic_candidate は使えません")
                        break
                else:
                    # print("負になる拘束がなかったのでこれは使えます")
                    batch_initial_conditions_filterd.append(ic_candidate)

            if len(batch_initial_conditions_filterd) < n_req:
                continue

            else:
                # batch_initial_conditions_filterd = batch_initial_conditions_filterd[:n_req]
                batch_initial_conditions_filterd = batch_initial_conditions_filterd[
                    :n_req
                ]
                print(counter)
                break
        a = batch_initial_conditions_filterd
        batch_initial_conditions_filterd = torch.tensor(
            np.array([_a.numpy() for _a in a])
        ).double()

        return batch_initial_conditions_filterd

    def my_generator_2(
        *args,
        **kwargs,
    ):
        if "study" in globals():
            if len(study.best_trials) > 0:
                # print(study.best_trials)
                normalized_parameter_list = []
                for _trial in study.best_trials:
                    values = torch.tensor(list(_trial.params.values())).double()
                    n_values = normalize(
                        values, torch.tensor([[0, 0], [1, 2 * np.pi]]).double()
                    )
                    print(n_values)
                    normalized_parameter_list.append([n_values.numpy()])
                return torch.tensor(normalized_parameter_list).double()
        print("init")
        return (torch.tensor([[[0.5, 0.51]]]).double(),)  # normalised

    constraints_dict = dict(
        # inequality_constraints=inequality_constraints,
        # equality_constraints=equality_constraints,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        # ic_generator=gen_batch_initial_conditions,
        # ic_generator=my_generator,
        ic_generator=my_generator_2,
        # batch_initial_conditions=torch.tensor([[[0.5, 0.51]]]).double(),  # normalised
        sequential=True,
        # gen_candidates=gen_candidates_scipy,
    )
    return constraints_dict


def objective(trial: optuna.Trial):
    r = trial.suggest_float("r", 0, 1)
    theta = trial.suggest_float("theta", 0, 2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def ineq_cns(r, theta: torch.Tensor):
    # print(r, theta)
    return -np.pi + theta  # theta >= np.pi is feasible


def do_mock(constraints_dict):
    original_fun = optuna_integration.botorch.optimize_acqf

    options = dict()

    options.update({"batch_limit": 1})  # options for optimize_acqf
    options.update(
        dict(  # options for gen_candidates_scipy()
            # method='SLSQP',
            method="COBYLA",  # dict 形式の constraints を扱えるのは SLSQP か COBYLA しかない
            # with_grad=None,  # dont use grad(SLSQP では jac が ゼロになって即終了することがある...が、botorch の実装が with_grad 前提なのでここは変えられない（文法エラーになる）
            # disp=True,
        )
    )

    overwritten_and_constrainted_fun = NonOverwritablePartial(
        original_fun,
        **dict(
            # options={"batch_limit": 1, "maxiter": 200, "nonnegative": True},
            options=options,
            **constraints_dict,
        ),
        **(
            ic_gen_kwargs := dict(
                nonliner_constraints=constraints_dict[
                    "nonlinear_inequality_constraints"
                ],
            )
        ),
    )
    optuna_integration.botorch.optimize_acqf = overwritten_and_constrainted_fun


if __name__ == "__main__":

    constraints_dict = get_constraint_dict()

    do_mock(constraints_dict)

    sampler = BoTorchSampler(independent_sampler=None, n_startup_trials=0)

    study = optuna.create_study(sampler=sampler, directions=["minimize"] * 2)

    study.enqueue_trial(dict(r=0.5, theta=np.pi+0.1))

    study.optimize(
        objective,
        n_trials=20,
    )

    for trial in study.best_trials:
        print(trial.values)

# values=[-0.14742013181182087, -0.6150628452458238]
# values=[-0.9999999999999961, 8.742277580776199e-08]
