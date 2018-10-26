

class TSLite:
    def __init__(self, name, w=1, b=0):
        self.name = name
        self.w = w
        self.b = b

    def __add__(self, n):
        if isinstance(n, int):
            return TSLite(self.name, self.w, self.b + n)
        elif isinstance(n, TSLite):
            return [self, n]
        else:
            assert False

    def __mul__(self, n):
        if isinstance(n, int):
            return TSLite(self.name, self.w * n, self.b)
        else:
            assert False

    def __div__(self, n):
        if isinstance(n, int):
            return TSLite(self.name, self.w / n, self.b)
        else:
            assert False

    def __eq__(self, d):
        return self.name == d.name and self.w == d.w and self.b == d.b

    def __repr__(self):
        s = f'({self.name}*{self.w}+{self.b})'
        return s

