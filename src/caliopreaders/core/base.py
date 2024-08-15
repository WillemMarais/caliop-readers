from __future__ import annotations

import os
import attr
import copy
from pathlib import Path
from collections import deque
from yaml import load, FullLoader

from caliopreaders.utilities.general import get_dirP_of__file__

from typing import cast, List, Optional


@attr.s(auto_attribs=True)
class CALIOPConfig:
    """CALIOP config."""

    count_to_photon_count_scale_flt: float
    """Science digitizer count to photon count scaling."""
    pmt_nr_dynodes_int: int = 13
    """The number of PMT dynodes."""
    pmt_dynode_gain_flt: float = 2.962
    """The PMT gain per dynode."""
    skip_datasets: Optional[List[str]] = cast(List[str], attr.field())
    """The data arrays to skip in the dataset."""

    @classmethod
    def crt_from_cfg(cls, cfg_dct: dict) -> CALIOPConfig:
        cfg_dct = copy.deepcopy(cfg_dct)

        return CALIOPConfig(
            **cfg_dct
        )

    @skip_datasets.default  # type: ignore
    def default_skip_datasets(self):
        return list()


class CALIOPBase:
    """CALIOP abstract reader class."""

    def __init__(self,
                 in_fileP: Optional[Path] = None,
                 cfg_fileP: Optional[Path] = None):
        """

        Parameters
        ----------
        in_fileP_str: str
            The level-0 input file.
        cfg_fileP_str: str
            The config file path to where the configuration of the reader is archived.
        """

        if cfg_fileP is None:
            # Set the default file path to the config file.
            cfg_fileP = \
                Path(
                    get_dirP_of__file__(__file__),
                    '..',
                    'config',
                    'caliop.yaml'
                )

        self.in_fileP = cast(Path, in_fileP)
        self.cfg_fileP = cast(Path, cfg_fileP)

    @staticmethod
    def load_config(cfg_fileP: Optional[Path] = None) -> dict:
        """Load the YAML config file and return as dictionary.

        Returns
        -------
        yaml_cfg_dct: dict
            The YAML configuration as a dictionary.
        """

        if cfg_fileP is None:
            yaml_cfg_dct = dict()
        else:
            with open(cfg_fileP, 'r') as file_obj:
                yaml_cfg_dct = load(file_obj, Loader=FullLoader)

            # Go through the config file section, and expand all the environmental variables
            if 'config_files' in yaml_cfg_dct:
                for cfg_fileP_str in yaml_cfg_dct['config_files'].keys():
                    yaml_cfg_dct['config_files'][cfg_fileP_str] = \
                        os.path.expandvars(yaml_cfg_dct['config_files'][cfg_fileP_str])

        return yaml_cfg_dct

    @staticmethod
    def read_metadataparse(metadata_str: str) -> dict:
        """Parse the HDFv4-EOS metadata and return it as a dictionary."""

        metadata_str_lst = metadata_str.replace(' ', '').replace('\n\n', '\n').split('\n')

        metadata_dct = {}
        dct_queue_obj = deque()
        dct_queue_obj.append(metadata_dct)

        current_token_str = None
        for metadata_str in metadata_str_lst:
            if metadata_str == '':
                continue

            if '=' in metadata_str:
                token_str, value_str = metadata_str.split('=')
                current_token_str = token_str
            else:
                token_str = current_token_str
                value_str = metadata_str

            if token_str == 'GROUP':
                dct_queue_obj[-1][value_str] = dict()
                dct_queue_obj.append(dct_queue_obj[-1][value_str])

            elif token_str == 'END_GROUP':
                dct_queue_obj.pop()

            elif token_str == 'OBJECT':
                dct_queue_obj[-1][value_str] = dict()
                dct_queue_obj.append(dct_queue_obj[-1][value_str])

            elif token_str == 'END_OBJECT':
                dct_queue_obj.pop()

            elif token_str in ['CLASS', 'NUM_VAL', 'VALUE']:
                if token_str not in dct_queue_obj[-1]:
                    dct_queue_obj[-1][token_str] = value_str.strip('"')
                else:
                    dct_queue_obj[-1][token_str] += value_str.strip('"')

            else:
                continue

        return metadata_dct
