import os


def get_dirP_of__file__(__fileP_str__: str) -> str:
    return os.path.dirname(os.path.realpath(__fileP_str__))
