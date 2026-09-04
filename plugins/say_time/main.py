import time
from random import choice


def get_ctx() -> dict:
    """Get context from `stdin`"""
    import json
    import sys

    return json.loads(sys.stdin.readline() or "{}")


def submit(data: dict):
    """Submits data into IPC-channel"""
    from json import dumps

    print(dumps(data), flush=True)


def log(message: str, source: str = "say-time", level: str = "INFO"):
    """Logs info"""
    from json import dumps
    from sys import stderr

    stderr.write(
        dumps({"type": "log", "message": message, "source": source, "level": level})
        + "\n"
    )
    stderr.flush()


sounds = ["Time is: {time}", "It is {time}"]


def main():
    get_ctx()

    t = time.localtime()
    res = {
        "type": "say",
        "text": choice(sounds).format(time=f"{t.tm_hour}:{t.tm_min:02d}"),
    }
    submit(res)


if __name__ == "__main__":
    main()
