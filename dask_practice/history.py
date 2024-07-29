from time import sleep

import pandas as pd

from dask.distributed import Client, LocalCluster, ActorFuture, Lock

from logger import get_logger


logger = get_logger()

lock = Lock("record")


class History:
    _df = None

    def __init__(self):
        self._df = pd.DataFrame(columns=["a", "b", "c"])

    def record(self, values) -> ActorFuture or None:
        lock.acquire()
        logger.info('記録しています。')
        sleep(1)
        row = pd.DataFrame({k: [v] for k, v in zip(self._df.columns, values)})
        if len(self._df) == 0:
            self._df = row
        else:
            self._df = pd.concat([self._df, row]).reset_index(drop=True)
        lock.release()

    def get_df(self) -> ActorFuture or pd.DataFrame:
        return self._df
