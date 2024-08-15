import os
import logging
from pathlib import Path
from logging import Logger


def setup_logging_basic(
    level_str: str = "INFO",
    handler_obj_lst: list | None = None
):

    if handler_obj_lst is None:
        handler_obj_lst = []

    level_obj = getattr(logging, level_str)

    # Setup the logging
    if len(handler_obj_lst) > 0:
        logging.basicConfig(
            level=level_obj,
            handlers=handler_obj_lst,
            datefmt="%Y/%m/%d %H:%M",
            format="%(asctime)s: [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
        )

    else:
        logging.basicConfig(
            level=level_obj,
            datefmt="%Y/%m/%d %H:%M",
            format="%(asctime)s: [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
        )


def create_logger(
    file_path_str: str,
    root_module_str: str = "caliopreaders"
) -> Logger:
    # Decompose the file path
    file_path_str_lst = file_path_str.split(os.path.sep)
    # Find the root module index
    root_module_idx = file_path_str_lst.index(root_module_str)
    # Remove the extention from the file name
    module_name_str = Path(file_path_str_lst[-1]).stem
    # Create the module path
    module_path_str = ".".join(file_path_str_lst[root_module_idx:-1] + [module_name_str])

    return logging.getLogger(module_path_str)
