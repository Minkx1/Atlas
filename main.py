# main.py

import multiprocessing
import os
import warnings

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="multiprocessing.resource_tracker"
    )
    multiprocessing.freeze_support()  # fix C++ threads/processes errors
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from src import Atlas

    atlas = Atlas()
    atlas.start()
