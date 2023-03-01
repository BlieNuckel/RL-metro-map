def flat_map(outer_list: list) -> list:
    return [item for sublist in outer_list for item in sublist]
