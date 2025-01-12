from typing import Union, Optional, Callable, Tuple

def exists(val: Optional[object]) -> bool:
    """
    Check if a value exists (is not None).

    Args:
        val (Optional[object]): The value to check.

    Returns:
        bool: True if the value exists, False otherwise.
    """
    return val is not None

def default(val: Optional[object], default_val: object) -> object:
    """
    Return `val` if it exists, otherwise return `default_val`.

    Args:
        val (Optional[object]): The value to check.
        default_val (object): The default value to return if `val` is None.

    Returns:
        object: `val` if it exists, otherwise `default_val`.
    """
    return val if exists(val) else default_val

def always(val: object) -> Callable[..., object]:
    """
    Return a function that always returns `val`.

    Args:
        val (object): The value to always return.

    Returns:
        Callable[..., object]: A function that always returns `val`.
    """
    return lambda *args: val

def pair(t: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Convert `t` to a tuple if it isn't already.

    Args:
        t (Union[int, Tuple[int, int]]): The value to convert.

    Returns:
        Tuple[int, int]: A tuple of two integers.
    """
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numerator: int, denominator: int) -> bool:
    """
    Check if `numerator` is divisible by `denominator`.

    Args:
        numerator (int): The numerator.
        denominator (int): The denominator.

    Returns:
        bool: True if divisible, False otherwise.
    """
    return (numerator % denominator) == 0
