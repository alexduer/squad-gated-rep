from datetime import datetime


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
