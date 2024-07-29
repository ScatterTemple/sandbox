from typing import Optional
from parameter import ExpressionEvaluator

from logger import get_logger


logger = get_logger()


class FEM:

    def __init__(self):
        self.evaluator: Optional[ExpressionEvaluator] = None

    def update(self):
        logger.info('FEM を更新しています。')
        self.evaluator.get_variables()
        ...  # solve something
