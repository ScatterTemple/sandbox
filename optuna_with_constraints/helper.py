from typing import Optional, List, Tuple, Callable
from functools import partial

import numpy as np
import torch
from torch import Tensor
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.utils.transforms import unnormalize
from optuna.study import Study

# module to monkey patch
import optuna_integration


class NonOverwritablePartial(partial):
    def __call__(self, /, *args, **keywords):
        stored_kwargs = self.keywords
        keywords.update(stored_kwargs)
        return self.func(*self.args, *args, **keywords)


class OptunaBotorchWithParameterConstraintMonkeyPatch:

    def __init__(
        self,
        study: Study,
        optprob,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        nonlinear_inequality_constraints: Optional[List[Tuple[Callable, bool]]] = None,
    ):
        self.study = study
        self.optprob = optprob
        self.nonlinear_inequality_constraints = nonlinear_inequality_constraints
        self.additional_kwargs = dict(
            q=1,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            ic_generator=self.generate_initial_conditions,
        )

    def add_nonlinear_constraint(self, fun):
        bounds = [
            [p['lb'] for p in self.optprob.prm_list],
            [p['ub'] for p in self.optprob.prm_list]
        ]

        self.nonlinear_inequality_constraints = self.nonlinear_inequality_constraints or []
        self.nonlinear_inequality_constraints.append(
            (
                lambda x: fun(*(unnormalize(x, torch.tensor(bounds)).double())),
                True
            )
        )

        self.additional_kwargs.update(
            nonlinear_inequality_constraints=self.nonlinear_inequality_constraints
        )

    def generate_initial_conditions(
        self,
        *args,
        **kwargs,
    ):
        batch_initial_conditions_feasible = []

        # とても遅くなり、かつ、精度も悪くなっているかも。
        # n 回 batch_initial_conditions を行い、feasible のもののみ抽出
        counter = 0
        n = 2
        batch_initial_conditions = gen_batch_initial_conditions(*args, **kwargs)
        while True:
            counter += 1
            if counter > n:
                break

            # 各初期値提案について
            for ic_candidate in batch_initial_conditions:
                # すべての非線形拘束について
                for cns in self.nonlinear_inequality_constraints:
                    # ひとつでも拘束を満たしていなければ初期値提案の処理を抜ける
                    if cns[0](*ic_candidate) < 0:
                        break
                else:
                    # 初期値提案がすべての非線形拘束を満たしたら feasible に入れる
                    batch_initial_conditions_feasible.append(ic_candidate.numpy())

        # study の履歴からも initial conditions を作成。
        # 内部で acquisition function が更新されるので同じ値が提案されてもよい
        for trial in self.study.best_trials:
            bounds = [[], []]  # [[lower_values], [upper_values]]
            prm_names, values = list(trial.params.keys()), list(trial.params.values())
            for prm_name in prm_names:
                dist = trial.distributions[prm_name]
                bounds[0].append(dist.low)
                bounds[1].append(dist.high)
            bounds = np.array(bounds).astype(float)
            normalized_values = (np.array(values).astype(float) - bounds[0]) / (bounds[1] - bounds[0])
            batch_initial_conditions_feasible.append([normalized_values])

        # もしここで feasible なものがなければ、どうしようもない
        if len(batch_initial_conditions_feasible) == 0:
            raise RuntimeError("candidate 探索のための拘束を満たす初期値を提案できませんでした。")

        return torch.tensor(batch_initial_conditions_feasible).double()

    def do_monkey_patch(self):
        """optuna_integration.botorch には optimize_acqf に constraints を渡す方法が用意されていないので、モンキーパッチして渡す"""

        # reconstruct argument ``options`` for optimize_acqf
        options = dict()  # initialize

        # for nonlinear-constraint
        options.update(dict(batch_limit=1))

        # for gen_candidates_scipy()
        options.update(dict(method='COBYLA'))  # use COBYLA instead of SLSQP. This is the only method that can process dict format constraints.

        # make partial of optimize_acqf used in optuna_integration.botorch and replace to it.
        original_fun = optuna_integration.botorch.optimize_acqf
        overwritten_fun = NonOverwritablePartial(
            original_fun,
            options=options,
            **self.additional_kwargs,
        )
        optuna_integration.botorch.optimize_acqf = overwritten_fun
