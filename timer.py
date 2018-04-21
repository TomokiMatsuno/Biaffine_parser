import time


class Timer(object):
    def __init__(self):
        self.start = self.prev = time.time()

    def from_start(self):
        print(time.time() - self.start)

    def from_prev(self):
        now = time.time()
        print(now - self.prev)
        self.prev = now
