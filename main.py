#
# main.py
# Entrypoint for Atlas
#

import multiprocessing
import os
import warnings
from contextlib import suppress


def main(no_ui: bool = True) -> None:
    from atlas import Atlas

    atlas = Atlas(ignored_modules=["ui"] if no_ui else [])
    atlas.run()


if __name__ == "__main__":
    # supresses warning that can break Textual UI
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="multiprocessing.resource_tracker"
    )
    # silence C++ threads/processes errors
    multiprocessing.freeze_support()
    with suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

    # optmiziation and other improvements
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main(no_ui=True)
