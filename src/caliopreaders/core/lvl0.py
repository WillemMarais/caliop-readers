import os
import json
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from pyhdf.VS import VS  # NOTE: do not remove this line!
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF, HC
from datetime import datetime
from pyhdf.error import HDF4Error

from caliopreaders.core.base import CALIOPBase

from typing import List, Optional

# Setup logging
log_obj = logging.getLogger(__name__)


class CALIOPlevel0(CALIOPBase):
    """Reader for the CALIOP level-0 file."""

    def __init__(self,
                 lvl0_fileP_lst: List[Path],
                 cfg_fileP: Optional[Path] = None,
                 yaml_cfg_dct: Optional[dict] = None):
        """

        Parameters
        ----------
        lvl0_fileP_lst: List[Path]
            The level-0 input files.
        cfg_fileP: Path
            The config file path to where the configuration of the reader is
            archived.
        yaml_cfg_dct: dict
            The configuration dictionary.
        """

        super(CALIOPlevel0, self).__init__(None, cfg_fileP)

        self._lvl0_fileP_lst = lvl0_fileP_lst
        self._yaml_cfg_dct = yaml_cfg_dct

    def _l0_to_xr(self,
                  lvl0_fileP: Path,
                  tqdm_kwargs: Optional[dict] = None) -> xr.Dataset:
        """Convert the data to a xarray data object, written to `out_fileP_str`.

        Parameters
        ----------
        lvl0_fileP: Path
            The level-0 file path.
        tqdm_kwargs: dict
            Extra keyword parameters to tqdm.

        Returns
        -------
        xr.Dataset:
            The dataset.
        """

        if tqdm_kwargs is None:
            tqdm_kwargs = dict(desc='level-0 dataset: ')
        else:
            tqdm_kwargs = tqdm_kwargs | dict(desc='level-0 dataset: ')

        # Read the config file
        if self._yaml_cfg_dct is None:
            yaml_cfg_dct = \
                self.load_config(self.cfg_fileP)['Readers']['CALIOPlevel0'] \
                | self.load_config(self.cfg_fileP)['Readers']['CALIOPBase']
        else:
            yaml_cfg_dct = self._yaml_cfg_dct['Readers']['CALIOPlevel0']

        log_obj.info(f'Reading the level-0 file "{self.in_fileP}"')
        sd_obj = SD(str(lvl0_fileP), SDC.READ)

        # Create the data and coordinate dictionaries
        data_dct = dict()
        coord_dct = dict()

        for ds_name_str, ds_info_tpl in sd_obj.datasets().items():
            if ds_name_str == 'Profile_Time':
                if 'time_utc' in coord_dct:
                    err_str = 'The time coordinate has already been set.'
                    raise ValueError(err_str)

                # Get the profile time; seconds since 1993 January first.
                prfl_time_arr = sd_obj.select('Profile_Time').get().ravel()
                # Convert to datetime64
                prfl_time_dt_arr = (prfl_time_arr * 1e9).astype('timedelta64[ns]') \
                    + np.datetime64(datetime(1993, 1, 1))

                coord_dct['time_utc'] = prfl_time_dt_arr

            elif ds_name_str == 'Profile_TAI_Time':
                if 'time_utc' in coord_dct:
                    err_str = 'The time coordinate has already been set.'
                    raise ValueError(err_str)

                # Get the profile time; seconds since 1993 January first.
                prfl_time_arr = sd_obj.select('Profile_TAI_Time').get().ravel()
                # Convert to datetime64
                prfl_time_dt_arr = (prfl_time_arr * 1e9).astype('timedelta64[ns]') \
                    + np.datetime64(datetime(1993, 1, 1))

                coord_dct['time_utc'] = prfl_time_dt_arr

            elif ds_name_str in ['LIDAR_DATA_ALTITUDE', 'Lidar_Data_Altitude']:
                altitude_arr = sd_obj.select(ds_name_str).get().ravel()

                coord_dct['altitude'] = altitude_arr

            elif ds_name_str == 'Altitude_Raw_Data':
                altitude_arr = sd_obj.select('Altitude_Raw_Data').get().ravel()

                coord_dct['raw_altitude'] = altitude_arr

            elif ds_name_str == 'Altitude_532_Averaged':
                altitude_arr = sd_obj.select('Altitude_532_Averaged').get().ravel()

                coord_dct['altitude'] = altitude_arr

            elif ds_name_str in yaml_cfg_dct['skip_datasets']:
                continue

            else:
                # Get the number of dimensions
                nr_columns_int, nr_rows_int = ds_info_tpl[1]

                # Get the units attribute
                attrs_dct = {}
                try:
                    attrs_dct = sd_obj.select(ds_name_str).attributes()
                    attrs_dct.pop('format', None)

                except (AttributeError, HDF4Error):
                    pass

                # Check what values should be scaled (from km to meters)
                if ds_name_str in ['Range_To_Mean_Sea_Level', 'Backscatter_Start_Altitude']:
                    scale_int = 1
                    attrs_dct['units'] = 'km'

                else:
                    scale_int = 1

                if (nr_rows_int == 1) and (nr_columns_int == 1):
                    data_dct[ds_name_str] = (
                        ['time_utc'],
                        scale_int * sd_obj.select(ds_name_str).get().ravel() * np.ones(shape=(15, )),
                        attrs_dct
                    )

                elif nr_columns_int == 1:
                    data_dct[ds_name_str] = (
                        ['altitude'],
                        scale_int * sd_obj.select(ds_name_str).get().ravel(),
                        attrs_dct
                    )

                elif nr_rows_int == 1:
                    data_dct[ds_name_str] = (
                        ['time_utc'],
                        scale_int * sd_obj.select(ds_name_str).get().ravel(),
                        attrs_dct
                    )

                elif nr_rows_int == 4800:
                    data_dct[ds_name_str] = (
                        ['time_utc', 'raw_altitude'],
                        scale_int * sd_obj.select(ds_name_str).get(),
                        attrs_dct
                    )

                else:
                    data_dct[ds_name_str] = (
                        ['time_utc', 'altitude'],
                        scale_int * sd_obj.select(ds_name_str).get(),
                        attrs_dct
                    )

        try:
            if 'altitude' not in coord_dct:
                vs_obj = HDF(lvl0_fileP, HC.READ).vstart()
                metadata_vd_obj = vs_obj.attach('metadata')

                all_metadata_name_str_lst = []
                for metadata_tpl in metadata_vd_obj.fieldinfo():
                    all_metadata_name_str_lst.append(metadata_tpl[0])

                alt_arr_idx = all_metadata_name_str_lst.index('Lidar_Data_Altitudes')
                coord_dct['altitude'] = np.array(metadata_vd_obj[:][0][alt_arr_idx])

                vs_obj.end()

        except ValueError:
            warning_str = \
                f'Cannot find an altitude array in the file "{lvl0_fileP.name}"'
            log_obj.warning(warning_str)

        # Convert the global metadata into a json strings
        global_attrs_dct = dict()
        global_attrs_dct['level0_coremetadata'] = \
            json.dumps(self.read_metadataparse(sd_obj.attributes()['coremetadata']))
        global_attrs_dct['level0_archivemetadata'] = \
            json.dumps(self.read_metadataparse(sd_obj.attributes()['archivemetadata']))

        # Close the HDFv4 file
        sd_obj.end()

        # Create the dataset
        ds = xr.Dataset(data_dct, coords=coord_dct, attrs=global_attrs_dct)
        # Set necessary units
        # ds['time_utc'].attrs['units'] = 'nanosecond'
        ds['altitude'].attrs['units'] = 'km'

        return ds

    def to_xr(self,
              out_fileP: Optional[Path] = None,
              time_utc_slice: Optional[slice] = None,
              tqdm_kwargs: Optional[dict] = None):
        """Convert the data to a xarray dataset, written to `out_fileP_str`.

        Parameters
        ----------
        out_fileP_str: str
            The output file to write to, optional.
        time_utc_slice: slice
            The time-axis selection. This slice selection will be adjusted so that the number of
            columns are multiptles of 15.
        tqdm_kwargs: dict
            Extra keyword parameters to tqdm.

        Returns
        -------
        xr.Dataset:
            The dataset.
        """

        # Read the level1B files
        ds_lst = [
            self._l0_to_xr(fileP, tqdm_kwargs)
            for fileP in self._lvl0_fileP_lst
        ]

        if len(ds_lst) == 0:
            err_str = 'Could not find level-0 dataset.'
            raise FileNotFoundError(err_str)

        if len(ds_lst) > 1:
            # Reorder the list of datasets so that they are increasing in time
            time_dt_lst: List[np.datetime64] = []
            for ds in ds_lst:
                time_dt_lst.append(ds['time_utc'].data.min())
            sort_idx_arr = np.argsort(time_dt_lst)
            sort_ds_lst = []
            for sort_idx in sort_idx_arr:
                sort_ds_lst.append(ds_lst[sort_idx])

            # Concatenate the datasets
            log_obj.info(f'Concatenating {len(sort_ds_lst)} level-0 files')
            l0_ds = xr.concat(sort_ds_lst, dim='time_utc', combine_attrs='override')

        else:
            l0_ds = ds_lst[0]

        # Get the start indices of where the 15-profile intervals which are called frames
        frame_number_arr = l0_ds['Frame_Number'].data
        frame_start_idx_arr = np.where(np.abs(np.diff(frame_number_arr)) > 0)[0] + 1

        # Selection a portion of the data over the time axis
        if time_utc_slice is not None:
            # Get the indices where the time-utc slice is
            start_time_idx = np.argmin(np.abs(l0_ds['time_utc'].data - time_utc_slice.start))
            stop_time_idx = np.argmin(np.abs(l0_ds['time_utc'].data - time_utc_slice.stop))

            # Adjust the start and stop indices so that they are aligned with the data frames
            start_time_idx = frame_start_idx_arr[np.argmin(np.abs(frame_start_idx_arr - start_time_idx))]
            stop_time_idx = frame_start_idx_arr[np.argmin(np.abs(frame_start_idx_arr - stop_time_idx))]

            adj_time_utc_slice = slice(
                l0_ds['time_utc'].data[start_time_idx],
                l0_ds['time_utc'].data[stop_time_idx - 1]
            )

            l0_ds = l0_ds.loc[{'time_utc': adj_time_utc_slice}]

        # HACK: Solve this in a different way
        l0_ds['time_utc'].attrs.pop('units', None)

        l0_ds.altitude.attrs |= {
            "long_name": "Altitude (MSL)",
            "units": "km"
        }

        # Create the output file
        if out_fileP is not None:
            out_fileP.parent.mkdir(exist_ok=True, parents=True)
            l0_ds.to_netcdf(out_fileP)

        return l0_ds


    # !!!!!!!!!!!!! DO NOT DELETE !!!!!!!!!!!!!
    # Parameters
    # ----------
    # out_fileP_str: str
    #     The output file to write to, optional.
    # time_utc_slice: slice
    #     The time-axis selection. This slice selection will be adjusted so that the number of
    #     columns are multiptles of 15.
    # tqdm_kwargs: dict
    #     Extra keyword parameters to tqdm.

    # # Get the start indices of where the 15-profile intervals which are called frames
    # frame_number_arr = ds['Frame_Number'].data
    # frame_start_idx_arr = np.where(np.abs(np.diff(frame_number_arr)) > 0)[0] + 1

    # # Selection a portion of the data over the time axis
    # if time_utc_slice is not None:
    #     # Get the indices where the time-utc slice is
    #     start_time_idx = np.argmin(np.abs(ds['time_utc'].data - time_utc_slice.start))
    #     stop_time_idx = np.argmin(np.abs(ds['time_utc'].data - time_utc_slice.stop))

    #     # Adjust the start and stop indices so that they are aligned with the data frames
    #     start_time_idx = frame_start_idx_arr[np.argmin(np.abs(frame_start_idx_arr - start_time_idx))]
    #     stop_time_idx = frame_start_idx_arr[np.argmin(np.abs(frame_start_idx_arr - stop_time_idx))]

    #     adj_time_utc_slice = slice(
    #         ds['time_utc'].data[start_time_idx],
    #         ds['time_utc'].data[stop_time_idx - 1]
    #     )

    #     ds = ds.loc[{'time_utc': adj_time_utc_slice}]

    # # # Correct for the un-normalized perpendicular background counts
    # # if 'Perpendicular_Amplifier_Gain_532' in ds:
    # #     log_obj.warning('Correcting for the unnormalized perpendicular background counts')
    # #     ds['Perpendicular_Background_Monitor_532'].data /= ds['Perpendicular_Amplifier_Gain_532'].data

    # # HACK: Solve this in a different way
    # ds['time_utc'].attrs.pop('units')

    # # Create the output file
    # if out_fileP_str is not None:
    #     os.makedirs(os.path.dirname(out_fileP_str), exist_ok=True)
    #     ds.to_netcdf(out_fileP_str)

    # return ds
