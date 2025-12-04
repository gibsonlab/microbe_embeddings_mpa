from contextlib import contextmanager
import time


@contextmanager
def timer(label=None, enabled=True):
    if enabled:
        start = time.perf_counter()
        yield  # code inside the "with" block runs here
        end = time.perf_counter()
        duration = end - start
        if label:
            print(f"[{label}] {duration:.6f} seconds")
        else:
            print(f"Took {duration:.6f} seconds")
    else:
        yield