class ProgressBar:
    def __init__(self, total=100, msg="progress:", bar_length=30):
        self.total = total
        self.msg = msg
        self.current = 0
        self.bar_length = bar_length

    def update(self, progress=None):
        if progress is None:
            self.current += 1
        else:
            self.current = progress
        print("\r", end="")

        bar = ""
        for i in range(self.bar_length):
            bar += "#" if i < self.current * self.bar_length // self.total else "-"
        print(f"[{bar}] - {self.msg} ({self.current}/{self.total}) ", end="")

    def conclude(self):
        print()