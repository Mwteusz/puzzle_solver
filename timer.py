import time
class Timer:
    def __init__(self):
        self.start_time = time.time()
    def stop(self,round_to=2):
        return round(time.time() - self.start_time,round_to)

    def print(self, message="",round_to=2):
        if message != "":
            message = f"[{message}]"
        print(f"--- {message} elapsed time: {self.stop(round_to=round_to)} seconds ---")
        return