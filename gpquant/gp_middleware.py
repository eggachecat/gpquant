import ctypes


class Middleware:
    def __init__(self, path_to_lib):
        self.lib = ctypes.cdll.LoadLibrary(path_to_lib)

    def get_function(self, func_attr):
        return getattr(self.lib, func_attr)

    def execute(self, func_attr, *args):
        func = self.get_function(func_attr)
        return func(*args)
