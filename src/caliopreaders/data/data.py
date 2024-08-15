"""Data for the simulations."""

import os
import pooch
from pooch import Pooch
from pathlib import Path
from yaml import load, FullLoader

from caliopreaders.config import sim_data_cfg_fileP
from caliopreaders.reporting import create_logger

log_obj = create_logger(__file__)


def expandvars_in_dict(obj, env_dct: dict) -> dict:
    """Go through all the items in a dictionary recursively. For every string that is in the
    dictionary, the function `os.path.expandvars` is applied on the string.

    Parameters
    ----------
    obj: object
        The dictionary or an item of the dictionary.
    """

    if isinstance(obj, dict) is True:
        for key_obj, value_obj in obj.items():
            obj[key_obj] = expandvars_in_dict(value_obj, env_dct)

    elif isinstance(obj, list) is True:
        for idx in range(len(obj)):
            obj[idx] = expandvars_in_dict(obj[idx], env_dct)

    elif isinstance(obj, tuple) is True:
        _obj = []
        for idx in range(len(obj)):
            _obj.append(expandvars_in_dict(obj[idx], env_dct))
        obj = tuple(_obj)

    elif isinstance(obj, str) is True:
        obj = os.path.expandvars(obj)

        for env_key_str, env_val_str in env_dct.items():
            if f'${env_key_str}' in obj:
                obj = obj.replace(f'${env_key_str}', env_val_str)

    return obj  # type: ignore


def create_sim_file_registery(
    archive_dirP: Path | None = None
) -> Pooch:
    """Create the simulation file registery.

    Parameters
    ----------
    archive_dirP: Path, optional
        Where the simulation files are
        archived. Default is ~/data_local/caliop-denoising.

    Returns
    -------
    pooch.Pooch:
        The pooch registery.
    """

    # Load the pooch config
    with open(sim_data_cfg_fileP, "r") as file_obj:
        cfg_dct = load(
            file_obj,
            Loader=FullLoader
        )

    # Expand any environmental variables
    cfg_dct = expandvars_in_dict(
        cfg_dct, dict()
    )

    if archive_dirP is not None:
        archive_dirP = archive_dirP.expanduser()
    else:
        archive_dirP = Path(
            cfg_dct["path_dct"]["local"]
        ).expanduser()

    log_obj.info(
        "Simulation file registery root "
        f"directory path: {archive_dirP}."
    )

    cfg_dct.pop("path_dct")
    cfg_dct["path"] = archive_dirP
    sim_file_registery = pooch.create(
        **cfg_dct
    )

    return sim_file_registery
