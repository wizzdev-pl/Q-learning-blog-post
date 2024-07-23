from collections import deque
import random


class ReplayMemory:
    """
    Memory for Experience Replay
    """

    def __init__(self, max_len: int):
        self.memory = deque([], maxlen=max_len)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size: int = 1):
        # Return k unique random elements
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
