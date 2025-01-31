from python.base import Context, Compiler


class Config:
    def __init__(self):
        # Initialized during import
        self.ctx: Context = None
        self.compiler: Compiler = None


config = Config()
