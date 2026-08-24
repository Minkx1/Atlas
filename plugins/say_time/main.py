def main(origin: str):
    import time

    t = time.localtime(time.time())
    s = f"{t.tm_hour}:{t.tm_min}"

    print("!", s, end="", flush=True)
