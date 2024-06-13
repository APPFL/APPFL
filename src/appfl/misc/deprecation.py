import warnings
import functools

# A set to keep track of warnings that have already been shown
_emitted_warnings = set()

def deprecated(reason='', silent=False):
    """
    This is a decorator which can be used to mark functions and classes
    as deprecated. It will result in a warning being emitted the first time
    the function or class is used, unless silent is True.
    """
    def decorator(obj):
        if isinstance(obj, type):
            # The obj is a class
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                if not silent and obj.__name__ not in _emitted_warnings:
                    warnings.warn(
                        f"{obj.__name__} is deprecated: {reason}",
                        category=DeprecationWarning,
                        stacklevel=2
                    )
                    _emitted_warnings.add(obj.__name__)
                orig_init(self, *args, **kwargs)

            obj.__init__ = new_init
            return obj
        else:
            # The obj is a function
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                if not silent and obj.__name__ not in _emitted_warnings:
                    warnings.warn(
                        f"{obj.__name__} is deprecated: {reason}",
                        category=DeprecationWarning,
                        stacklevel=2
                    )
                    _emitted_warnings.add(obj.__name__)
                return obj(*args, **kwargs)
            return wrapper
    return decorator

warnings.simplefilter('always', DeprecationWarning)