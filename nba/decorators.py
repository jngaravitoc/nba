import functools
import importlib

def requires_library(lib_name, fallback=None):
    def decorator(func):
        try:
            importlib.import_module(lib_name)
            return func
        except ImportError:
            if fallback is not None:
                return fallback
            else:
                @functools.wraps(func)
                def missing_lib_wrapper(*args, **kwargs):
                    raise ImportError(f"The function '{func.__name__}' requires the '{lib_name}' library.")
                return missing_lib_wrapper
    return decorator
