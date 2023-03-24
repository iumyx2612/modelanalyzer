import math


__all__ = [
    'is_perfect_square'
]


def is_perfect_square(number: int) -> bool:
    sq_root = int(math.sqrt(number))
    return (sq_root * sq_root) == number
