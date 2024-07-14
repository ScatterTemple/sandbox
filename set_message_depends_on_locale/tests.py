from message import encoding, Msg


if __name__ == '__main__':
    print(Msg.HELLO)  # こんにちは！

    with open('hello.csv', 'w') as f:
        f.write(Msg.HELLO)  # こんにちは！ を保存

    import pandas as pd

    print(pd.read_csv('hello.csv', encoding=encoding))
