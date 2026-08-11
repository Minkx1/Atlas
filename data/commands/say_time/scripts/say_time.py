import time


def main() -> str:
    return f"{time.strftime('%H:%M', time.localtime(time.time()))}"
