import ctypes

lib = ctypes.cdll.LoadLibrary('MathFuncsDll.dll')

dll_add = getattr(lib, "?Multiply@MyMathFuncs@MathFuncs@@SANNN@Z")
dll_add.restype = ctypes.c_double
print(dll_add(ctypes.c_double(2), ctypes.c_double(3)))