from python.xavier import Array


class Compiler:
    def compile(self, arr: Array):
        raise NotImplementedError("Abstract class.")


class Context:
    def __init__(self):
        raise NotImplementedError("Abstract class.")
