from typing import Iterable, TypeVar

T = TypeVar("T")


def flat_map(outer_list: Iterable[Iterable[T]]) -> list[T]:
    return [item for sublist in list(outer_list) for item in sublist]
