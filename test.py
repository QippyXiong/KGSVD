class O:
    def __init__(self):
        print('O init')


class A:
    def __init__(self):
        print('A init')

class B(O):
    def __init__(self):
        print('B init')

class C(A, B):
    def __init__(self):
        super(A, self).__init__()

if __name__ == '__main__':
    c = C()
    a = (
    "bbbbb"
    "ccccc"
    )
    print(a)

