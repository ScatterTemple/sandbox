import inspect
from collections import OrderedDict
from graphlib import TopologicalSorter


class ExpressionEvaluator:
    def __init__(self):
        self.parameters = {}
        self.expressions = {}
        self.dependencies = {}
        self.evaluation_order = []

    def add_parameter(self, name, value):
        self.parameters[name] = value
        self.dependencies[name] = set()

    def add_expression(self, name, fun):
        self.expressions[name] = fun
        # funの引数名を取得し、それを依存関係として設定する
        params = inspect.signature(fun).parameters
        self.dependencies[name] = set(params)

    def resolve(self):
        ts = TopologicalSorter(self.dependencies)
        self.evaluation_order = list(ts.static_order())

    def evaluate(self):
        # 順番に処理していく
        for variable in self.evaluation_order:
            # variable が (parameter でなく) expression であれば
            if variable in self.expressions:
                # 実行する
                args = {
                    param: self.parameters[param]
                    for param in self.dependencies[variable]
                }
                self.parameters[variable] = self.expressions[variable](**args)

    def get_value(self, name) -> float:
        return self.parameters.get(name, None)

    def get_variables(self) -> dict:
        return {name: self.get_value(name) for name in self.evaluation_order}


if __name__ == "__main__":
    # 使用例
    evaluator = ExpressionEvaluator()

    # 変数の追加
    evaluator.add_parameter("a", 2)
    evaluator.add_parameter("b", 3)

    # 式の追加（例：c = a + b, d = c * 2）
    evaluator.add_expression("c", lambda a, b: a + b)
    evaluator.add_expression("d", lambda c: c * 2)

    # 評価
    evaluator.resolve()
    evaluator.evaluate()

    # 結果の取得
    print('=====')
    print(f"a = {evaluator.get_value('a')}")
    print(f"b = {evaluator.get_value('b')}")
    print(f"c = {evaluator.get_value('c')}")
    print(f"d = {evaluator.get_value('d')}")

    # 変数の更新
    evaluator.add_parameter("a", 4)
    evaluator.add_parameter("b", 5)

    # 評価
    evaluator.evaluate()

    # 結果の取得
    print('=====')
    print(f"a = {evaluator.get_value('a')}")
    print(f"b = {evaluator.get_value('b')}")
    print(f"c = {evaluator.get_value('c')}")
    print(f"d = {evaluator.get_value('d')}")
