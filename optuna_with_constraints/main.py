from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import optuna
import botorch
from optuna.integration.botorch import BoTorchSampler

from helper import OptunaBotorchWithParameterConstraintMonkeyPatch

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.NumericsWarning)


@dataclass
class OptProb:
    prm_list: Optional[list] = field(default_factory=lambda: [])

    def add_parameter(self, name, init, lb, ub):
        self.prm_list.append(
            dict(
                name=name,
                init=float(init),
                lb=float(lb),
                ub=float(ub),
            )
        )

    def objective(self, trial: optuna.trial.Trial):
        x = []
        for p in self.prm_list:
            x.append(trial.suggest_float(p['name'], p['lb'], p['ub']))
        y0 = x[0] * np.cos(x[1])
        y1 = x[0] * np.sin(x[1])
        return y0, y1

    def constraint(self, r, theta):
        return - r + 0.5  # >= 0


if __name__ == "__main__":

    # 最適化問題を設定
    opt = OptProb()
    opt.add_parameter('r', 0.2, 0, 1)
    opt.add_parameter('theta', 0, 0, 2*3.141592)

    # アルゴリズムを指定
    sampler = BoTorchSampler(independent_sampler=None, n_startup_trials=0)

    # スタディを定義
    study = optuna.create_study(sampler=sampler, directions=["minimize"] * 2)

    # 初期値を指定
    study.enqueue_trial(dict(r=0.2, theta=0.))

    # モンキーパッチ実行
    mp = OptunaBotorchWithParameterConstraintMonkeyPatch(
        study,
        opt
    )
    mp.add_nonlinear_constraint(opt.constraint)
    mp.do_monkey_patch()

    # 実行
    study.optimize(
        opt.objective,
        n_trials=10,
    )

    for trial in study.best_trials:
        print(trial.values)

