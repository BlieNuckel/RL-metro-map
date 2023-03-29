from src.models import Coordinates2d, Stop


def line_overlap(lines: dict[Coordinates2d, str], position: Coordinates2d) -> bool:
    return position in lines.keys()


def stop_overlap(stops: dict[Coordinates2d, Stop], position: Coordinates2d) -> bool:
    return position in stops.keys()


def any_overlap(lines: dict[Coordinates2d, str], stops: dict[Coordinates2d, Stop], position: Coordinates2d) -> bool:
    return line_overlap(lines, position) or stop_overlap(stops, position)
