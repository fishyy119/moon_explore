import time


class MyTimer:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.records = []

    def checkpoint(self, label=""):
        now = time.time()
        duration = now - self.last_time
        total = now - self.start_time
        self.records.append((label, duration, total))
        print(f"[{label}] {duration:.3f}s (Total: {total:.3f}s)")
        self.last_time = now
