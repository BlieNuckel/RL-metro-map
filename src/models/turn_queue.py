from collections import deque


class TurnQueue:
    def __init__(self, max_len: int) -> None:
        self.__turns = deque([False for _ in range(max_len)], maxlen=0)

    def non_turn(self):
        self.__turns.append(False)

    def turn(self):
        self.__turns.append(True)

    def num_of_turns(self) -> int:
        return len([turn for turn in self.__turns if turn])
