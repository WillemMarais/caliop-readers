import os
import json
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from pyhdf.VS import VS, VD  # NOTE: do not remove this line!
from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF, HC
from datetime import datetime
from scipy.interpolate import interp1d

from caliopreaders.core.base import CALIOPBase

from typing import cast, Any, Dict, List, Optional, Tuple

# Setup logging
log_obj = logging.getLogger(__name__)


# TODO: Disentangle the downloading for L1B data from this class. Create the a separate class that downloads
# the L1B data that covers the level-0 file, and this class prepares the level-1B data for the level-0 data.
class CALIOPlevel1B(CALIOPBase):
    """Reader for the CALIOP level-1B file."""

    # Make a list of the ancillary meteorological data that should be interpolated on the level-0 altitude
    # coordinate
    met_ds_name_str_lst = [
        'Molecular_Number_Density',
        'Ozone_Number_Density',
        'Temperature',
        'Pressure',
        'Relative_Humidity'
    ]

    def __init__(self,
                 lvl1B_fileP_lst: List[Path],
                 lvl0_ds: Optional[xr.Dataset] = None,
                 cfg_fileP: Optional[Path] = None):
        """

        Parameters
        ----------
        lvl1B_fileP_str_lst: List[str]
            The file paths to the level-1B files that cover the level-0 file if any.
        lvl0_ds: xr.Dataset, optional
            The CALIOP level-0 dataset.
        cfg_fileP: Path, optional
            The config file path to where the configuration of the reader is archived.
        """

        super(CALIOPlevel1B, self).__init__(None, cfg_fileP)

        self._lvl1B_fileP_lst = lvl1B_fileP_lst

        if lvl0_ds is not None:
            if np.mod(lvl0_ds['time_utc'].size, 15) != 0:
                err_str = 'The level-0 time axis MUST be divisible by 15'
                raise ValueError(err_str)

        self._lvl0_ds = lvl0_ds

    def _get_interpolated_met_data(self,
                                   alt_arr: np.ndarray,
                                   sd_obj: SD,
                                   metadata_vd_obj: VD,
                                   prfl_time_dt_arr: np.datetime64) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        """Get the ancillary meteorological data interpolated over the level-0 altitude coordinate axis.

        Parameters
        ----------
        alt_arr: np.ndarray
            The altitude array.
        sd_obj: SD
            The L1B scientific dataset.
        metadata_vd_obj: VD
            The L1B metadata.
        prfl_time_dt_arr: np.ndarray
            The temporal coordinate axis of the data.

        Returns
        -------
        met_data_dct: Dict[str, np.ndarray]
            The interpolated ancillary meteorological data. The dictionary has the following keys:
                - Molecular_Number_Density
                - Ozone_Number_Density
                - Temperature
                - Pressure
                - Relative_Humidity
                - molecular_rayleigh_backscatter_cross_section_532
                - molecular_rayleigh_backscatter_cross_section_1064
                - molecular_rayleigh_extinction_cross_section_532
                - molecular_rayleigh_extinction_cross_section_1064

        met_data_units_dct: Dict[str, str]
            The units of the ancillary meteorological data.
        """

        # Get the indices of the specific metadata
        find_metadata_name_str_lst = [
            'Met_Data_Altitudes',
            'Rayleigh_Extinction_Cross-section_532',
            'Rayleigh_Extinction_Cross-section_1064',
            'Rayleigh_Backscatter_Cross-section_532',
            'Rayleigh_Backscatter_Cross-section_1064',
            'Ozone_Absorption_Cross-section_532',
            'Ozone_Absorption_Cross-section_1064'
        ]

        all_metadata_name_str_lst = []
        for metadata_tpl in metadata_vd_obj.fieldinfo():
            all_metadata_name_str_lst.append(metadata_tpl[0])

        find_metadata_name_idx_lst = []
        for find_metadata_name_str in find_metadata_name_str_lst:
            find_metadata_name_idx_lst.append(all_metadata_name_str_lst.index(find_metadata_name_str))

        # Get the metadata values
        metadata_dct = dict()
        for find_metadata_name_str, find_metadata_name_idx in zip(find_metadata_name_str_lst,
                                                                  find_metadata_name_idx_lst):

            if find_metadata_name_str == 'Met_Data_Altitudes':
                scale_flt = 1
                metadata_dct[find_metadata_name_str] = \
                    scale_flt * np.array(metadata_vd_obj[:][0][find_metadata_name_idx])

            else:
                scale_flt = 1.0
                metadata_dct[find_metadata_name_str] = scale_flt * metadata_vd_obj[:][0][find_metadata_name_idx]

        # Interpolate the ancillary meteorological data over the level-0 altitude coordinate
        met_data_dct = dict()
        for met_ds_name_str in self.met_ds_name_str_lst:
            if met_ds_name_str in ['Temperature', 'Ozone_Number_Density']:
                def transform_fnc(_x):  # type: ignore
                    return _x

                def inv_transform_fnc(_x):  # type: ignore
                    return _x

            else:
                def transform_fnc(_x):
                    with np.errstate(invalid='ignore'):  # type: ignore
                        return np.log(_x)

                def inv_transform_fnc(_x):
                    return np.exp(_x)

            sd_ds_obj = sd_obj.select(met_ds_name_str)

            # Get the met data
            met_data_arr = sd_ds_obj.get()

            # Replace fill values with NaN values, and apply transform on data
            met_data_arr[np.where(met_data_arr == -999)] = np.nan
            met_data_arr = transform_fnc(met_data_arr)

            # If there are any missing values in the met data, interpolate over it
            if bool(np.any(np.isnan(met_data_arr))) is True:
                met_data_da = \
                    xr.DataArray(
                        np.fliplr(met_data_arr),
                        coords=[  # type: ignore
                            prfl_time_dt_arr,   # type: ignore
                            np.flipud(metadata_dct['Met_Data_Altitudes'])
                        ],
                        dims=['time_utc', 'met_altitude_msl']
                    )
                met_data_arr = \
                    np.fliplr(met_data_da.interpolate_na(dim='time_utc', method='cubic').data)

            # Interpolate over the lidar altitude coordinate axis
            flip_met_data_alt_arr = np.flipud(metadata_dct['Met_Data_Altitudes'])
            flip_met_data_arr = np.fliplr(met_data_arr)
            flip_altitude_msl_arr = np.flipud(alt_arr)

            interp_meta_data_arr = np.zeros(shape=(flip_met_data_arr.shape[0], flip_altitude_msl_arr.size))
            for prfl_idx in range(interp_meta_data_arr.shape[0]):
                not_nan_idx_arr = np.where(np.logical_not(np.isnan(flip_met_data_arr[prfl_idx, :])))[0]

                interp_meta_data_arr[prfl_idx, :] = \
                    np.flipud(
                        interp1d(
                            flip_met_data_alt_arr[not_nan_idx_arr],
                            flip_met_data_arr[prfl_idx, not_nan_idx_arr],
                            kind='linear',
                            fill_value='extrapolate'  # type: ignore
                        )(flip_altitude_msl_arr))

            met_data_dct[met_ds_name_str] = inv_transform_fnc(interp_meta_data_arr)

            if bool(np.any(np.isnan(met_data_dct[met_ds_name_str]))) is True:
                err_str = 'There are NaN values in the interpolated '\
                    f'ancillary meteorological dataset "{met_ds_name_str}"'
                raise ValueError(err_str)

        # From the molecular density, compute the molecular Rayleigh backscatter and extinction cross-sections per
        # unit volume
        met_data_dct['molecular_rayleigh_backscatter_cross_section_532'] = \
            1.0e+3 \
            * metadata_dct['Rayleigh_Backscatter_Cross-section_532'] \
            * met_data_dct['Molecular_Number_Density']

        met_data_dct['molecular_rayleigh_backscatter_cross_section_1064'] = \
            1.0e+3 \
            * metadata_dct['Rayleigh_Backscatter_Cross-section_1064'] \
            * met_data_dct['Molecular_Number_Density']

        met_data_dct['molecular_rayleigh_extinction_cross_section_532'] = \
            1.0e+3 \
            * metadata_dct['Rayleigh_Extinction_Cross-section_532'] \
            * met_data_dct['Molecular_Number_Density']

        met_data_dct['molecular_rayleigh_extinction_cross_section_1064'] = \
            1.0e+3 \
            * metadata_dct['Rayleigh_Extinction_Cross-section_1064'] \
            * met_data_dct['Molecular_Number_Density']

        # From the ozone density, compute the ozone cross-sections per unit volume
        met_data_dct['ozone_extinction_cross_section_532'] = \
            1.0e+3 \
            * metadata_dct['Ozone_Absorption_Cross-section_532'] \
            * met_data_dct['Ozone_Number_Density']

        met_data_dct['ozone_extinction_cross_section_1064'] = \
            1.0e+3 \
            * metadata_dct['Ozone_Absorption_Cross-section_1064'] \
            * met_data_dct['Ozone_Number_Density']

        met_data_units_dct = {
            'Molecular_Number_Density': 'molecules / m^3',
            'Ozone_Number_Density': 'molecules / m^3',
            'Temperature': 'degrees C',
            'Pressure': 'millibars',
            'Relative_Humidity': '%',
            'molecular_rayleigh_backscatter_cross_section_532': '1/km 1/sr',
            'molecular_rayleigh_backscatter_cross_section_1064': '1/km 1/sr',
            'molecular_rayleigh_extinction_cross_section_532': '1/km',
            'molecular_rayleigh_extinction_cross_section_1064': '1/km',
            'ozone_extinction_cross_section_532': '1/km',
            'ozone_extinction_cross_section_1064': '1/km'
        }

        return met_data_dct, met_data_units_dct

    @staticmethod
    def _process_l1b_datarray(data_arr: np.ndarray,
                              attrs_dct: dict,
                              ds_name_str: str,
                              out_coord_dct: Dict[str, Any],
                              out_data_dct: Dict[str, Tuple[List[str], np.ndarray, dict]]):
        """Process the SD data array as specified by `ds_name_str`.

        Parameters
        ----------
        data_arr: np.ndarray
            The data array.
        attrs_dct: dict
            The attributes of the data array.
        ds_name_str: str
            The data array to process.
        out_coord_dct: Dict[str, np.ndarray]
            The output coordinate dictionary.
        out_data_dct: Dict[str, Tuple[List[str], np.ndarray, dict]]
            The output data dictionary.
        """

        if ds_name_str == 'Profile_Time':
            # Get the profile time; seconds since 1993 January first.
            prfl_time_arr = data_arr.ravel()
            # Convert to datetime64
            prfl_time_dt_arr = \
                cast(np.timedelta64, (prfl_time_arr * 1e9).astype('timedelta64[ns]')) \
                + np.datetime64(datetime(1993, 1, 1))

            out_coord_dct['time_utc'] = prfl_time_dt_arr

        elif ds_name_str == 'Surface_Wind_Speeds':
            out_coord_dct['wind_direction'] = [  # type: ignore
                'eastward',
                'northward'
            ]

            out_data_dct['Surface_Wind_Speeds'] = (
                ['time_utc', 'wind_direction'],
                data_arr,
                attrs_dct
            )

        elif ds_name_str in [
            'Spacecraft_Position',
            'Spacecraft_Velocity',
            'Spacecraft_Attitude',
            'Spacecraft_Attitude_Rate'
        ]:

            out_coord_dct['ECR'] = [  # type: ignore
                'x-axis',
                'y-axis',
                'z-axis'
            ]

            if ds_name_str == 'Spacecraft_Velocity':
                scale_int = 1
                attrs_dct['units'] = 'km/s'

            else:
                scale_int = 1

            out_data_dct[ds_name_str] = (
                ['time_utc', 'ECR'],
                scale_int * data_arr,
                attrs_dct
            )

        else:
            # Get the number of dimensions
            nr_columns_int, nr_rows_int = data_arr.shape

            # Check what values should be scaled (from km to meters)
            if ds_name_str in [
                'Tropopause_Height',
                'Surface_Elevation',
                'GMAO_Surface_Elevation',
                'Surface_Altitude_Shift',
                'Spacecraft_Altitude',
                'Spacecraft_Position'
            ]:

                scale_int = 1
                attrs_dct['units'] = 'km'

            elif ds_name_str in [
                'Calibration_Constant_532',
                'Calibration_Constant_Uncertainty_532',
                'Calibration_Constant_1064',
                'Calibration_Constant_Uncertainty_1064'
            ]:
                scale_int = 1
                attrs_dct['units'] = 'km^3 * sr * count'

            elif ds_name_str in [
                'Total_Attenuated_Backscatter_532',
                'Perpendicular_Attenuated_Backscatter_532',
                'Attenuated_Backscatter_1064'
            ]:
                scale_int = 1
                attrs_dct['units'] = '1/km 1/sr'

            elif ds_name_str in [
                'Parallel_Amplifier_Gain_532',
                'Perpendicular_Amplifier_Gain_532',
                'Amplifier_Gain_1064'
            ]:
                scale_int = 1
                attrs_dct['units'] = 'V/V'

            elif ds_name_str in [
                'Latitude',
                'Longitude',
                'Off_Nadir_Angle'
            ]:
                scale_int = 1
                attrs_dct['units'] = 'deg'

            elif ds_name_str in [
                'Parallel_Background_Monitor_532',
                'Perpendicular_Background_Monitor_532'
            ]:
                scale_int = 1
                attrs_dct['units'] = 'count'

            elif ds_name_str in [
                'Laser_Energy_532',
                'Laser_Energy_1064'
            ]:
                scale_int = 1
                attrs_dct['units'] = 'J'

            else:
                scale_int = 1

            if nr_columns_int == 1:
                out_data_dct[ds_name_str] = (
                    ['altitude'],
                    scale_int * data_arr.ravel(),
                    attrs_dct
                )

            elif nr_rows_int == 1:
                out_data_dct[ds_name_str] = (
                    ['time_utc'],
                    scale_int * data_arr.ravel(),
                    attrs_dct
                )

            else:
                out_data_dct[ds_name_str] = (
                    ['time_utc', 'altitude'],
                    scale_int * data_arr,
                    attrs_dct
                )

    def _l1b_to_xr(self, level1B_fileP: Path, incld_met_data_bl: bool) -> xr.Dataset:
        """Read the L1B and output as xr.Dataset."""

        # Read the config file
        yaml_cfg_dct = self.load_config(self.cfg_fileP)['Readers']['CALIOPlevel1B']

        # Open the HDFv4 file
        log_obj.info(f'Reading the level-1B file {level1B_fileP}')
        sd_obj = SD(str(level1B_fileP), SDC.READ)
        vs_obj = HDF(str(level1B_fileP), HC.READ).vstart()
        # Get the metadata
        metadata_vd_obj = vs_obj.attach('metadata')

        # Create the data and coordinate dictionaries
        data_dct = dict()
        coord_dct = dict()
        global_attrs_dct = dict()

        # Get the temporal coordinate axis
        ds_name_str = 'Profile_Time'
        try:
            attrs_dct = {
                'units': sd_obj.select(ds_name_str).attribute()['units']  # type: ignore
            }
        except AttributeError:
            attrs_dct = dict()

        # Get the altitude coordinate
        all_metadata_name_str_lst = []
        for metadata_tpl in metadata_vd_obj.fieldinfo():
            all_metadata_name_str_lst.append(metadata_tpl[0])

        alt_arr_idx = all_metadata_name_str_lst.index('Lidar_Data_Altitudes')
        alt_arr = np.array(metadata_vd_obj[:][0][alt_arr_idx])

        alt_da = \
            xr.DataArray(
                alt_arr,
                coords={
                    'altitude': alt_arr
                },
                dims=['altitude'],
                attrs={
                    'long_name': 'Altitude',
                    'units': 'km'
                }
            )

        # Add the altitude to the coordinate array
        coord_dct['altitude'] = alt_da

        self._process_l1b_datarray(sd_obj.select(ds_name_str).get(),
                                   attrs_dct,
                                   'Profile_Time',
                                   coord_dct,
                                   data_dct)

        # Get the list of datasets to skip
        skip_datasets_str_lst = yaml_cfg_dct['skip_datasets']

        if incld_met_data_bl is True:
            # Get the ancillary meteorological data
            met_data_dct, met_data_units_dct = \
                self._get_interpolated_met_data(
                    alt_arr,
                    sd_obj,
                    metadata_vd_obj,
                    coord_dct['time_utc']
                )
        else:
            met_data_dct = dict()
            met_data_units_dct = dict()

            skip_datasets_str_lst += self.met_ds_name_str_lst

        # Process the different data arrays
        ignore_ds_name_str_lst = ['Profile_Time'] + skip_datasets_str_lst + list(met_data_dct.keys())
        for ds_name_str in sd_obj.datasets().keys():
            if ds_name_str in ignore_ds_name_str_lst:
                continue

            # Get the units attribute
            try:
                attrs_dct = {
                    'units': sd_obj.select(ds_name_str).attribute()['units']  # type: ignore
                }
            except AttributeError:
                attrs_dct = dict()

            self._process_l1b_datarray(
                sd_obj.select(ds_name_str).get(),
                attrs_dct,
                ds_name_str,
                coord_dct,
                data_dct
            )

        # Add ancillary meteorological data arrays
        for met_ds_name_str in met_data_dct.keys():
            attrs_dct = {
                'units': met_data_units_dct[met_ds_name_str]
            }

            self._process_l1b_datarray(
                met_data_dct[met_ds_name_str],
                attrs_dct,
                met_ds_name_str,
                coord_dct,
                data_dct
            )

        # Convert the global metadata into a json strings
        global_attrs_dct['level1B_coremetadata'] = \
            json.dumps(self.read_metadataparse(sd_obj.attributes()['coremetadata']))
        global_attrs_dct['level1B_archivemetadata'] = \
            json.dumps(self.read_metadataparse(sd_obj.attributes()['archivemetadata']))

        # Close the HDFv4 file
        vs_obj.end()
        sd_obj.end()

        # Create the dataset
        ds = xr.Dataset(data_dct, coords=coord_dct, attrs=global_attrs_dct)

        return ds

    def to_xr(self,
              out_fileP: Optional[Path] = None,
              incld_met_data_bl: bool = True) -> xr.Dataset:
        """Convert the data to a xarray dataset, written to `out_fileP_str`.

        Parameters
        ----------
        out_fileP: Path
            The output file to write to, optional.
        incld_met_data_bl: bool
            If True, include MET data in the level-1B.

        Returns
        -------
        xr.Dataset:
            The dataset.
        """

        # Read the level1B files
        ds_lst = [
            self._l1b_to_xr(fileP, incld_met_data_bl)
            for fileP in self._lvl1B_fileP_lst
        ]

        if len(ds_lst) == 0:
            err_str = 'Could not find level-1B files for the given level-0 dataset'
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
            log_obj.info(f'Concatenating {len(sort_ds_lst)} level-1B files')
            l1b_ds = xr.concat(sort_ds_lst, dim='time_utc', combine_attrs='override')

        else:
            l1b_ds = ds_lst[0]

        if self._lvl0_ds is not None:
            log_obj.info(f'Truncating the level-1B file(s)')

            # Find the time index on the L1B time axis where the level-0 dataset starts. Here
            # we assume that the level-0 and level-1B datasets have the same time-axis.
            # Note: The level-0 and level-1B times axis sometimes differ by +-250ns;
            start_time_idx = \
                np.argmin(np.abs(
                    l1b_ds['time_utc'].data - self._lvl0_ds['time_utc'].data.min()
                ))

            # stop_time_idx = start_time_idx + self._lvl0_ds['time_utc'].data.size
            stop_time_idx = \
                np.argmin(np.abs(
                    l1b_ds['time_utc'].data - self._lvl0_ds['time_utc'].data.max()
                ))

            # Adjust the stop_time_idx if necessary
            loc_dct = {
                'time_utc': slice(
                    l1b_ds['time_utc'].data[start_time_idx],
                    l1b_ds['time_utc'].data[stop_time_idx]
                )
            }
            if l1b_ds['time_utc'].loc[loc_dct].size > self._lvl0_ds['time_utc'].data.size:
                stop_time_idx -= \
                    l1b_ds['time_utc'].loc[loc_dct].size \
                    - self._lvl0_ds['time_utc'].data.size

                loc_dct = {
                    'time_utc': slice(
                        l1b_ds['time_utc'].data[start_time_idx],
                        l1b_ds['time_utc'].data[stop_time_idx]
                    )
                }
            elif l1b_ds['time_utc'].loc[loc_dct].size < self._lvl0_ds['time_utc'].data.size:
                stop_time_idx += \
                    self._lvl0_ds['time_utc'].data.size \
                    - l1b_ds['time_utc'].loc[loc_dct].size

                loc_dct = {
                    'time_utc': slice(
                        l1b_ds['time_utc'].data[start_time_idx],
                        l1b_ds['time_utc'].data[stop_time_idx]
                    )
                }

            # Truncate the L1B dataset
            l1b_ds = l1b_ds.loc[loc_dct]

            # Use the level-0 time
            l1b_ds = l1b_ds.assign_coords(time_utc=self._lvl0_ds.time_utc)

            # Report statistics of the time axis differences between the level0 and level1B datasets
            time_diff_arr = (self._lvl0_ds['time_utc'].data - l1b_ds['time_utc'].data).astype(float)
            info_str = 'The time difference statistics between level0 and level1B datasets: \n'
            info_str += '    min: {:.2f} [ns]\n'.format(time_diff_arr.min())
            info_str += '    max: {:.2f} [ns]\n'.format(time_diff_arr.max())
            info_str += '    mean: {:.2f} [ns]\n'.format(time_diff_arr.mean())
            info_str += '    std: {:.2f} [ns]'.format(time_diff_arr.std())
            log_obj.info(info_str)

        l1b_ds.altitude.attrs |= {
            "long_name": "Altitude (MSL)",
            "units": "km"
        }

        # Create the output file
        if out_fileP is not None:
            os.makedirs(os.path.dirname(out_fileP), exist_ok=True)
            l1b_ds.to_netcdf(out_fileP)

        return l1b_ds


def create_dt_map(l1b_fileP_str_lst: List[str]) -> Dict[Tuple[np.datetime64, np.datetime64], str]:
    """Create a mapping between L1B files and the time interval that each L1B file spans."""

    dt_to_file_map_dct: Dict[Tuple[np.datetime64, np.datetime64], str] = dict()
    for l1b_fileP_str in l1b_fileP_str_lst:
        # Get the metadata
        sd_obj = SD(l1b_fileP_str, SDC.READ)
        cr_md_dct = CALIOPlevel1B.read_metadataparse(sd_obj.attributes()['coremetadata'])
        sd_obj.end()

        # Get the start and end data
        start_dt = \
            np.datetime64(
                cr_md_dct['INVENTORYMETADATA']['TEMPORALINFORMATION']['START_DATE']['VALUE'][:-1]
            )
        stop_dt = \
            np.datetime64(
                cr_md_dct['INVENTORYMETADATA']['TEMPORALINFORMATION']['STOP_DATE']['VALUE'][:-1]
            )

        dt_to_file_map_dct[(start_dt, stop_dt)] = l1b_fileP_str

    return dt_to_file_map_dct


def from_dt_map_get_fileP(dt_to_file_map_dct: Dict[Tuple[np.datetime64, np.datetime64], str],
                          dt: np.datetime64) -> str:
    """From the mapping between L1B files and the time interval that each L1B file spans,
    return the file path which matches the given datetime."""

    trgt_fileP_str = None
    for dt_tpl, fileP_str in dt_to_file_map_dct.items():
        if (dt_tpl[0] <= dt) and (dt <= dt_tpl[1]):
            trgt_fileP_str = fileP_str
            break

    if trgt_fileP_str is None:
        raise FileNotFoundError('Could not find a L1B file for datetime ' + str(dt))

    return trgt_fileP_str
