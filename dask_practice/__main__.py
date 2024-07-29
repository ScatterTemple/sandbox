from typing import Optional

from dask.distributed import Client, LocalCluster, ActorFuture, Lock
import pandas as pd
import random
from time import sleep

from optimizer import Optimizer
from history import History
from parameter import ExpressionEvaluator
from logger import get_logger


logger = get_logger()


def add_worker(client, worker_name):
    from subprocess import Popen, DEVNULL
    import sys

    Popen(
        f"{sys.executable} -m dask worker {client.scheduler.address} --nthreads 1 --nworkers 1 --name {worker_name} --no-nanny",
        shell=True,
        stderr=DEVNULL,
        stdout=DEVNULL,
    )


def main():
    # 計算用クラスの準備
    opt = Optimizer()

    # パラメータ管理クラスの定義
    evaluator = ExpressionEvaluator()

    # パラメータの追加
    evaluator.add_variable("a", 2)
    evaluator.add_variable("b", 3)

    # 式の追加（例：c = a + b, d = c * 2）
    evaluator.add_expression('c', lambda a, b: a + b)
    evaluator.add_expression('d', lambda c: c * 2)

    # パラメータ・式の依存関係の評価
    evaluator.resolve()

    # 計算用クラスターの用意
    logger.info("計算用クラスターを立ち上げます。")
    cluster = LocalCluster(
        name="sample cluster",
        n_workers=3,
        threads_per_worker=1,
        processes=True,
    )
    client = Client(
        address=cluster,
    )

    # 計算用ワーカーのホストネームを取得
    calc_workers = [w.worker_address for w in cluster.workers.values()]

    # 記録用ワーカーの用意
    logger.info("記録用ワーカーを立ち上げます。")
    add_worker(client, "additional_worker")

    # 記録アクターの起動
    history_future = client.submit(
        History,
        actor=True,
        workers="additional_worker",
        allow_other_workers=False,
    )
    history: History = history_future.result()

    # 記録アクターと変数管理クラスを計算クラスに割り当て
    opt.history = history
    opt.evaluator = evaluator

    # 計算用ワーカーに計算を投げる
    logger.info("計算を始めます。")
    worker_indices = list(range(3))
    worker_features = client.map(
        opt.calc,
        worker_indices,
        workers=calc_workers,
        allow_other_workers=False,
        pure=False,
    )

    # 計算終了待ち
    client.gather(worker_features)

    # 結果
    df = history.get_df().result()
    logger.info("計算が終了しました。")
    logger.info(df)

    # 終了
    logger.info("クラスターを終了します。")
    client.cancel(history_future, force=True)
    workers = client.scheduler_info()["workers"]
    target_worker = None
    for k, v in workers.items():
        if v["id"] == "additional_worker":
            target_worker = k
    client.retire_workers(
        workers=[target_worker],
        close_workers=True,
    )
    cluster.close()
    client.close()
    client.shutdown()


if __name__ == "__main__":
    main()
