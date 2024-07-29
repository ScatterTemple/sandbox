from time import sleep
import datetime

import pandas as pd
import optuna
from optuna.distributions import FloatDistribution

from dask.distributed import Client, LocalCluster, ActorFuture, Lock

from logger import get_logger


logger = get_logger()

lock = Lock("record")


class History:
    _df = None

    def __init__(self):
        str_time = datetime.datetime.now().strftime("%M%S")

        # メインデータ
        self._df = pd.DataFrame(columns=["a", "b", "c"])

        # 可視化のための optuna スタディ
        # self.study = optuna.create_study(
        #     storage="sqlite:///history_" + str_time + ".db",
        #     sampler=None,
        #     pruner=None,
        #     study_name="history",
        #     direction=None,
        #     load_if_exists=False,
        #     directions=None,
        # )

    def record(self, values) -> ActorFuture or None:
        lock.acquire()
        # メインデータへの記録
        logger.info("記録しています。")
        sleep(1)
        row = pd.DataFrame({k: [v] for k, v in zip(self._df.columns, values)})
        if len(self._df) == 0:
            self._df = row
        else:
            self._df = pd.concat([self._df, row]).reset_index(drop=True)

        # optuna スタディへの記録
        # kwargs = dict(
        #     state=optuna.trial.TrialState.COMPLETE,
        #     params={k: v for k, v in zip(h.prm_names, row[h.prm_names])},
        #     distributions={k: FloatDistribution(0, 1) for k in h.prm_names},
        #     user_attrs=None,
        #     system_attrs=None,
        #     intermediate_values=None,
        # )
        # if len(h.obj_names) == 1:
        #     kwargs.update(dict(value=row[h.obj_names].values[0]))
        # else:
        #     kwargs.update(dict(values=row[h.obj_names].values))
        # trial = optuna.create_trial(**kwargs)
        # self.study.add_trial(trial)

        lock.release()

    def get_df(self) -> ActorFuture or pd.DataFrame:
        return self._df
