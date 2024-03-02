
from typing import Any, List, Union, Optional, Sequence, Mapping, Literal, Dict

### Singleton Class ###

class MetaSingleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    """
    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(
                MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(object, metaclass=MetaSingleton):
    """
    Base class for implementing the Singleton pattern.
    """

    def __init__(self):
        super(Singleton, self).__init__()


