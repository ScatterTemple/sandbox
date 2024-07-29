from time import sleep
from typing import Optional

import numpy as np

from history import History
from parameter import ExpressionEvaluator
from logger import get_logger


logger = get_logger()


class Optimizer:

    def __init__(self):
        self.history: Optional[History] = None
        self.evaluator: Optional[ExpressionEvaluator] = None

    def calc(self, idx):
        logger.info("計算しています。")

        for i in range(3):
            self.evaluator.add_variable("a", np.random.rand())
            self.evaluator.add_variable("b", np.random.rand())
            self.evaluator.evaluate()
            c = self.evaluator.get_value("c")
            d = self.evaluator.get_value("d")
            self.history.record([idx, c, d])
