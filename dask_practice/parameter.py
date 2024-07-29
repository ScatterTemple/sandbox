import inspect
from graphlib import TopologicalSorter


class ExpressionEvaluator:
    def __init__(self):
        self.variables = {}
        self.expressions = {}
        self.dependencies = {}
        self.evaluation_order = []

    def add_variable(self, name, value):
        self.variables[name] = value
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
                    param: self.variables[param]
                    for param in self.dependencies[variable]
                }
                self.variables[variable] = self.expressions[variable](**args)

    def get_value(self, name):
        return self.variables.get(name, None)


if __name__ == "__main__":
    # 使用例
    evaluator = ExpressionEvaluator()

    # 変数の追加
    evaluator.add_variable("a", 2)
    evaluator.add_variable("b", 3)

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
    evaluator.add_variable("a", 4)
    evaluator.add_variable("b", 5)

    # 評価
    evaluator.evaluate()

    # 結果の取得
    print('=====')
    print(f"a = {evaluator.get_value('a')}")
    print(f"b = {evaluator.get_value('b')}")
    print(f"c = {evaluator.get_value('c')}")
    print(f"d = {evaluator.get_value('d')}")
