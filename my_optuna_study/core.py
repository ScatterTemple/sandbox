import datetime

import numpy as np
import pandas as pd
import optuna


class History:

    def __init__(self):
        self.df = pd.DataFrame()
        self.prm_names = []
        self.obj_names = []

    def construct(self, length=1, n_var=3, n_obj=1):
        d = {}
        d_p = {f"prm{i}": np.random.rand(length) for i in range(n_var)}
        d_o = {f"obj{i}": np.random.rand(length) for i in range(n_obj)}
        d.update(d_p)
        d.update(d_o)
        self.df = pd.DataFrame(d)
        self.prm_names = list(d_p.keys())
        self.obj_names = list(d_o.keys())

    def append(self):
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    np.random.rand(1, len(self.df.columns)), columns=self.df.columns
                ),
            ],
        )

    def __str__(self):
        return self.df.__str__()
