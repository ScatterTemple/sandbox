from time import sleep
from typing import Optional, Iterable

import numpy as np

from history import History
from parameter import ExpressionEvaluator
from logger import get_logger
from fem import FEM


logger = get_logger()


class Optimizer:

    def __init__(self):
        self.history: Optional[History] = None
        self.evaluator: Optional[ExpressionEvaluator] = None
        self.fem: Optional[FEM] = None

    def f(self, x: Iterable[float]):
        self.evaluator.add_parameter("a", np.random.rand())
        self.evaluator.add_parameter("b", np.random.rand())

    def calc(self, idx):
        logger.info("計算しています。")

        for i in range(3):
            self.evaluator.add_parameter("a", np.random.rand())
            self.evaluator.add_parameter("b", np.random.rand())
            self.evaluator.evaluate()
            c = self.evaluator.get_value("c")
            d = self.evaluator.get_value("d")
            self.history.record([idx, c, d])
