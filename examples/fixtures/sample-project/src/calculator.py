# (c) JFrog Ltd. (2026)

"""Simple calculator module - deliberately missing tests and with a known bug."""


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    # Bug: no zero-division guard
    return a / b
