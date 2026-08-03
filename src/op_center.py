#
# op_center.py
# Center Of Operations: processes commands from STT
#

from .config import DATA_DIR, OS_NAME, cfg
import .ui

class Operator:
    def __init__(self) -> None:
        pass

    def operate(self, text: str) -> None: ...
