# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from insardev_toolkit import datagrid
from .PRM_gmtsar import PRM_gmtsar

class PRM(datagrid, PRM_gmtsar):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from typing import Any, List, Union
    
    int_types = ['num_valid_az', 'num_rng_bins', 'num_patches', 'bytes_per_line', 'good_bytes_per_line', 'num_lines','SC_identity']

    @staticmethod
    def to_numeric_or_original(val: str|float|int) -> str|float|int:
        if isinstance(val, str):
            try:
                float_val = float(val)
                return float_val
            except ValueError:
                return val
        return val

    @staticmethod
    def from_list(prm_list: list) -> "PRM":
        """
        Convert a list of parameter and value pairs to a PRM object.

        Parameters
        ----------
        prm_list : list
            A list of PRM strings.

        Returns
        -------
        PRM
            A PRM object.
        """
        from io import StringIO
        prm = StringIO('\n'.join(prm_list))
        return PRM._from_io(prm)

    @staticmethod
    def from_str(prm_string: str) -> "PRM":
        """
        Convert a string of parameter and value pairs to a PRM object.

        Parameters
        ----------
        prm_string : str
            A PRM string.

        Returns
        -------
        PRM
            A PRM object.
        """
        from io import StringIO
        if isinstance(prm_string, bytes):
            # for cases like
            #return PRM.from_str(os.read(pipe2[0],int(10e6))
            prm_string = prm_string.decode('utf-8')
        # for cases like
        # return PRM.from_str(os.read(pipe2[0],int(10e6).decode('utf8'))
        prm = StringIO(prm_string)
        return PRM._from_io(prm)

    @staticmethod
    def from_file(prm_filename: str) -> "PRM":
        """
        Convert a PRM file of parameter and value pairs to a PRM object.

        Parameters
        ----------
        prm_filename : str
            The filename of the PRM file.

        Returns
        -------
        PRM
            A PRM object.
        """
        #data = json.loads(document)
        prm = PRM._from_io(prm_filename)
        prm.filename = prm_filename
        return prm

    @staticmethod
    def _from_io(prm: str) -> "PRM":
        """
        Read parameter and value pairs from IO stream to a PRM object.

        Parameters
        ----------
        prm : IO stream
            The IO stream.

        Returns
        -------
        PRM
            A PRM object.
        """
        import pandas as pd

        df = pd.read_csv(prm, sep=r'\s+=\s+', header=None, names=['name', 'value'], engine='python').set_index('name')
        df['value'] = df['value'].map(PRM.to_numeric_or_original)

        return PRM(df)

    def __init__(self, prm: Union["PRM", pd.DataFrame, None]=None):
        """
        Initialize a PRM object.

        Parameters
        ----------
        prm : PRM or pd.DataFrame, optional
            The PRM object or DataFrame to initialize from. Default is None.

        Returns
        -------
        None
        """
        import pandas as pd

        # Initialize an empty DataFrame if prm is None
        if prm is None:
            _prm = pd.DataFrame(columns=['name', 'value'])
        elif isinstance(prm, pd.DataFrame):
            _prm = prm.reset_index()
        else:
            _prm = prm.df.reset_index()

        # Convert values to numeric where possible, keep original value otherwise
        _prm['value'] = _prm['value'].map(PRM.to_numeric_or_original)

        # Set the DataFrame for the PRM object
        self.df = _prm[['name', 'value']].drop_duplicates(keep='last').set_index('name')
        self.filename = None
        self.orbit_df = None  # In-memory orbit data (replaces LED file)

    def __eq__(self, other: "PRM") -> bool:
        """
        Compare two PRM objects for equality.

        Parameters
        ----------
        other : PRM
            The other PRM object to compare with.

        Returns
        -------
        bool
            True if the PRM objects are equal, False otherwise.
        """
        return isinstance(self, PRM) and self.df == other.df

    def __str__(self) -> str:
        """
        Return a string representation of the PRM object.

        Returns
        -------
        str
            The string representation of the PRM object.
        """
        return self.to_str()

    def __repr__(self) -> str:
        """
        Return a string representation of the PRM object for debugging.

        Returns
        -------
        str
            The string representation of the PRM object. If the PRM object was created from a file, 
            the filename and the number of items in the DataFrame representation of the PRM object 
            are included in the string.
        """
        if self.filename:
            return 'Object %s (%s) %d items\n%r' % (self.__class__.__name__, self.filename, len(self.df), self.df)
        else:
            return 'Object %s %d items\n%r' % (self.__class__.__name__, len(self.df), self.df)

    # use 'g' format for Python and numpy float values
    def set(self, prm: Union["PRM", None]=None, **kwargs) -> "PRM":
        """
        Set PRM values.

        Note: This method only copies DataFrame values, not orbit_df.
        To copy orbit_df, assign it separately: prm_copy.orbit_df = prm_orig.orbit_df

        Parameters
        ----------
        prm : PRM, optional
            The PRM object to set values from. Default is None.
        **kwargs
            Additional keyword arguments for setting individual values.

        Returns
        -------
        PRM
            The updated PRM object.
        """
        import numpy as np

        if isinstance(prm, PRM):
            for (key, value) in prm.df.itertuples():
                self.df.loc[key] = value
        elif prm is not None:
            raise Exception('Arguments is not a PRM object')
        for key, value in kwargs.items():
            self.df.loc[key] = value
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the PRM object to a DataFrame.

        Returns
        -------
        pd.DataFrame
            The DataFrame representation of the PRM object.
        """
        return self.df

    def to_file(self, filename: str) -> "PRM":
        """
        Save the PRM object to a PRM file.

        Parameters
        ----------
        prm : str
            The filename of the PRM file to save to.

        Returns
        -------
        PRM
            The PRM object.
        """
        self._to_io(filename)
        # update internal filename after saving with the new filename
        self.filename = filename
        return self

    def update(self, debug: bool=False) -> "PRM":
        """
        Save PRM file to disk.

        Parameters
        ----------
        debug : bool, optional
            Whether to enable debug mode. Default is False.

        Returns
        -------
        PRM
            The updated PRM object.
        """
        if self.filename is None:
            raise Exception('PRM is not created from file, use to_file() method instead')

        if debug:
            print ('DEBUG:', self)

        return self.to_file(self.filename)

    def to_str(self) -> str:
        """
        Convert the PRM object to a string.

        Returns
        -------
        str
            The PRM string.
        """
        return self._to_io()

    def _to_io(self, output: str|None=None) -> str:
        """
        Convert the PRM object to an IO stream.

        Parameters
        ----------
        output : IO stream, optional
            The IO stream to write the PRM string to. Default is None.

        Returns
        -------
        str
            The PRM string.
        """
        return self.df.reset_index().astype(str).apply(lambda row: (' = ').join(row), axis=1)\
            .to_csv(output, header=None, index=None)

    def sel(self, *args: str) -> "PRM":
        """
        Select specific PRM attributes and create a new PRM object.

        Parameters
        ----------
        *args : str
            The attribute names to select.

        Returns
        -------
        PRM
            The new PRM object with selected attributes.
        """
        return PRM(self.df.loc[[*args]])

    def __add__(self, other: Union["PRM", float, int]) -> "PRM":
        """
        Add two PRM objects or a PRM object and a scalar.

        Parameters
        ----------
        other : PRM or scalar
            The PRM object or scalar to add.

        Returns
        -------
        PRM
            The resulting PRM object after addition.
        """
        import pandas as pd
        if isinstance(other, PRM):
            prm = pd.concat([self.df, other.df])
            # drop duplicates
            prm = prm.groupby(prm.index).last()
        else:
            prm = self.df + other
        return PRM(prm)

    def __sub__(self, other: Union["PRM", float, int]) -> "PRM":
        """
        Subtract two PRM objects or a PRM object and a scalar.

        Parameters
        ----------
        other : PRM or scalar
            The PRM object or scalar to subtract.

        Returns
        -------
        PRM
            The resulting PRM object after subtraction.
        """
        import pandas as pd
        if isinstance(other, PRM):
            prm = pd.concat([self.df, other.df])
            # drop duplicates
            prm = prm.groupby(prm.index).last()
        else:
            prm = self.df - other
        return PRM(prm)

    def get(self, *args: str) -> Union[Any, List[Any]]:
        """
        Get the values of specific PRM attributes.

        Parameters
        ----------
        *args : str
            The attribute names to get values for.

        Returns
        -------
        Union[Any, List[Any]]
            The values of the specified attributes. If only one attribute is requested, 
            return its value directly. If multiple attributes are requested, return a list of values.
        """
        vals = [self.df.loc[[key]].iloc[0].values[0] for key in args]
        out = [int(v) if k in self.int_types else v for (k, v) in zip(args,vals)]
        if len(out) == 1:
            return out[0]
        return out

    def fix_aligned(self) -> "PRM":
        """
        Correction for the range and azimuth shifts of the re-aligned SLC images (fix_prm_params() in GMTSAR)
        """
        from scipy import constants
        # constant from GMTSAR code
        #SOL = 299792456.0
    
        delr = constants.speed_of_light / self.get('rng_samp_rate') / 2
        #delr = SOL / self.get('rng_samp_rate') / 2
        near_range = self.get('near_range') + \
            (self.get('st_rng_bin') - self.get('chirp_ext') + self.get('rshift') + self.get('sub_int_r') - 1)* delr
    
        SC_clock_start = self.get('SC_clock_start') + \
            (self.get('ashift') + self.get('sub_int_a')) / (self.get('PRF') * 86400.0) + \
            (self.get('nrows') - self.get('num_valid_az')) / (2 * self.get('PRF') * 86400.0)
    
        SC_clock_stop = SC_clock_start + \
            (self.get('num_valid_az') * self.get('num_patches')) / (self.get('PRF') * 86400.0)
    
        return self.set(near_range=near_range,
                        SC_clock_start=SC_clock_start,
                        SC_clock_stop=SC_clock_stop)

    # note: only one dimension chunked due to sequential file reading 
    def read_SLC_int(self) -> xr.Dataset:
        """
        Read SLC (Single Look Complex) data and compute the power of the signal.
        The method reads binary SLC data file, which contains alternating sequences of real and imaginary parts.
        It calculates the intensity of the signal and return it as a 2D numpy array.

        Returns
        -------
        xarray.DataArray
            2D array representing the power of the signal. The array shape corresponds to the dimensions of the SLC data.

        Notes
        -----
        This function uses a data factor (DFACT = 2.5e-07) from the GMTSAR code.
        The GMTSAR note indicates that the square of the intensity is used to match gips ihconv.
        The returned intensity data is flipped up-down ("shift data up if necessary") following the GMTSAR convention.

        Raises
        ------
        FileNotFoundError
            If the SLC file cannot be found.

        Example
        -------
        >>> import numpy as np
        >>> prm = PRM.from_file(filename)
        >>> amp = prm.read_SLC_int()
        """
        import xarray as xr
        import numpy as np
        import os

        prm = PRM.from_file(self.filename)
        slc_filename, xdim, ydim, rshift, ashift = prm.get('SLC_file', 'num_rng_bins', 'num_valid_az', 'rshift', 'ashift')

        dirname = os.path.dirname(self.filename)
        slc_filename = os.path.join(dirname, slc_filename)

        # read entire SLC file using memory mapping
        # SLC format: [real_0, imag_0, real_1, imag_1, real_2, imag_2, ...]
        slc_data = np.memmap(slc_filename, dtype=np.int16, mode='r')

        # extract real and imaginary parts
        total_pixels = ydim * xdim
        re = slc_data[0:2*total_pixels:2].reshape((ydim, xdim))
        im = slc_data[1:2*total_pixels:2].reshape((ydim, xdim))

        assert re.shape == (ydim, xdim), f'Originated re shape ({ydim},{xdim}), but got {re.shape}'
        assert im.shape == (ydim, xdim), f'Originated im width ({ydim},{xdim}), but got {im.shape}'

        coords = {'a': np.arange(ydim) + 0.5, 'r': np.arange(xdim) + 0.5}
        re = xr.DataArray(re, coords=coords, dims=['a', 'r']).rename('re')
        im = xr.DataArray(im, coords=coords, dims=['a', 'r']).rename('im')
        return xr.merge([re, im])

    def read_tops_params(self) -> dict:
        """
        Read TOPS-specific parameters from burst XML annotation for phase ramp computation.

        These parameters are required to compute the TOPS azimuthal phase ramp between
        adjacent bursts. The phase ramp formula is:

            phase = -π * kt * (η - η_ref)² - 2π * fnc * η

        where:
            kt = ka * ks / (ka - ks)  (effective Doppler rate)
            ks = 2 * v * fc * kψ / c  (steering rate contribution)
            ka = azimuthFmRate polynomial evaluated at slant range time
            fnc = Doppler centroid polynomial evaluated at slant range time
            η = azimuth time relative to burst center
            η_ref = reference time = -fnc/ka + fnc[0]/ka[0]

        Returns
        -------
        dict
            Dictionary containing:
            - azimuthSteeringRate: Antenna steering rate in degrees/second
            - azimuthFmRatePolynomial: FM rate coefficients [c0, c1, c2] at burst center time
            - azimuthFmRateT0: Reference slant range time for FM rate polynomial
            - azimuthFmRateAzimuthTime: Azimuth time for FM rate record
            - dcPolynomial: Doppler centroid coefficients [c0, c1, c2] at burst center time
            - dcT0: Reference slant range time for DC polynomial
            - dcAzimuthTime: Azimuth time for DC estimate record

        Notes
        -----
        The FM rate and DC polynomials are time-varying. This method selects the record
        closest to the burst center time, following the same approach as GMTSAR's
        make_s1a_tops.c dramp_dmod() function.

        The XML annotation file is derived from the input_file path by replacing .tiff with .xml.

        References
        ----------
        GMTSAR make_s1a_tops.c: https://github.com/gmtsar/gmtsar
        ESA Sentinel-1 TOPS SLC Deramping: https://sentinels.copernicus.eu/documents/247904/1653442/Sentinel-1-TOPS-SLC_Deramping
        """
        from datetime import datetime
        import xmltodict

        # Derive XML path from input_file (replace .tiff with .xml)
        input_file = self.get('input_file')
        xml_file = input_file.replace('measurement','annotation').replace('.tiff', '.xml')

        # Read annotation XML
        with open(xml_file) as fd:
            annotation = xmltodict.parse(fd.read())

        product = annotation['product']

        # Get azimuth steering rate (constant for entire burst)
        azimuth_steering_rate = float(product['generalAnnotation']['productInformation']['azimuthSteeringRate'])

        # Get azimuth time interval for computing burst center time
        dta = float(product['imageAnnotation']['imageInformation']['azimuthTimeInterval'])

        # Get lines per burst
        lpb = int(product['swathTiming']['linesPerBurst'])

        # Get burst azimuth time and compute burst center time
        burst_list = product['swathTiming']['burstList']['burst']
        if isinstance(burst_list, list):
            burst_info = burst_list[0]  # Single burst file has only one burst
        else:
            burst_info = burst_list
        burst_azimuth_time_str = burst_info['azimuthTime']
        burst_azimuth_time = datetime.strptime(burst_azimuth_time_str, '%Y-%m-%dT%H:%M:%S.%f')
        # Burst center time (as fractional day for comparison)
        t_brst = (burst_azimuth_time.hour * 3600 + burst_azimuth_time.minute * 60 +
                  burst_azimuth_time.second + burst_azimuth_time.microsecond / 1e6)
        t_brst += dta * lpb / 2.0  # Add half burst duration

        # Parse firstValidSample to get ksr (first valid line) and ker (last valid line)
        # These are needed for GMTSAR-style overlap boundary computation
        first_valid = burst_info['firstValidSample']
        if isinstance(first_valid, dict):
            fvs_str = first_valid['#text']
        else:
            fvs_str = first_valid
        fvs = [int(x) for x in fvs_str.split()]

        ksr = None  # First valid line index within burst
        ker = None  # Last valid line index within burst
        for j, flag in enumerate(fvs):
            if flag >= 0:
                if ksr is None:
                    ksr = j
                ker = j

        # Azimuth ANX time for computing overlap with adjacent bursts
        azimuth_anx_time = float(burst_info['azimuthAnxTime'])

        # Find azimuthFmRate record closest to burst center
        fm_rate_list = product['generalAnnotation']['azimuthFmRateList']['azimuthFmRate']
        if not isinstance(fm_rate_list, list):
            fm_rate_list = [fm_rate_list]

        best_fm_idx = 0
        best_fm_diff = float('inf')
        for i, fm_rate in enumerate(fm_rate_list):
            fm_time_str = fm_rate['azimuthTime']
            fm_time = datetime.strptime(fm_time_str, '%Y-%m-%dT%H:%M:%S.%f')
            fm_t = fm_time.hour * 3600 + fm_time.minute * 60 + fm_time.second + fm_time.microsecond / 1e6
            diff = abs(fm_t - t_brst)
            if diff < best_fm_diff:
                best_fm_diff = diff
                best_fm_idx = i

        fm_rate_record = fm_rate_list[best_fm_idx]
        fm_t0 = float(fm_rate_record['t0'])
        fm_azimuth_time = fm_rate_record['azimuthTime']

        # Parse FM rate polynomial - handle both old and new XML formats
        if 'azimuthFmRatePolynomial' in fm_rate_record:
            fm_poly_str = fm_rate_record['azimuthFmRatePolynomial']['#text'] if isinstance(fm_rate_record['azimuthFmRatePolynomial'], dict) else fm_rate_record['azimuthFmRatePolynomial']
            fm_poly = [float(x) for x in fm_poly_str.split()]
        else:
            # New format with c0, c1, c2 elements
            fm_poly = [float(fm_rate_record['c0']), float(fm_rate_record['c1']), float(fm_rate_record['c2'])]

        # Find dcEstimate record closest to burst center
        dc_list = product['dopplerCentroid']['dcEstimateList']['dcEstimate']
        if not isinstance(dc_list, list):
            dc_list = [dc_list]

        best_dc_idx = 0
        best_dc_diff = float('inf')
        for i, dc_est in enumerate(dc_list):
            dc_time_str = dc_est['azimuthTime']
            dc_time = datetime.strptime(dc_time_str, '%Y-%m-%dT%H:%M:%S.%f')
            dc_t = dc_time.hour * 3600 + dc_time.minute * 60 + dc_time.second + dc_time.microsecond / 1e6
            diff = abs(dc_t - t_brst)
            if diff < best_dc_diff:
                best_dc_diff = diff
                best_dc_idx = i

        dc_record = dc_list[best_dc_idx]
        dc_t0 = float(dc_record['t0'])
        dc_azimuth_time = dc_record['azimuthTime']

        # Parse DC polynomial
        dc_poly_str = dc_record['dataDcPolynomial']['#text'] if isinstance(dc_record['dataDcPolynomial'], dict) else dc_record['dataDcPolynomial']
        dc_poly = [float(x) for x in dc_poly_str.split()]

        return {
            'azimuthSteeringRate': azimuth_steering_rate,
            'azimuthFmRatePolynomial': fm_poly,
            'azimuthFmRateT0': fm_t0,
            'azimuthFmRateAzimuthTime': fm_azimuth_time,
            'dcPolynomial': dc_poly,
            'dcT0': dc_t0,
            'dcAzimuthTime': dc_azimuth_time,
            # GMTSAR-style overlap boundary parameters
            'linesPerBurst': lpb,
            'ksr': ksr,  # First valid line index within burst
            'ker': ker,  # Last valid line index within burst
            'azimuthAnxTime': azimuth_anx_time,  # Azimuth ANX time for overlap computation
            #'prf': float(product['imageAnnotation']['imageInformation']['azimuthFrequency']),
        }

    # my replacement function for GMT based robust 2D trend coefficient calculations:
    # gmt trend2d r.xyz -Fxyzmw -N1r -V
    # gmt trend2d r.xyz -Fxyzmw -N2r -V
    # gmt trend2d r.xyz -Fxyzmw -N3r -V
    # https://github.com/GenericMappingTools/gmt/blob/master/src/trend2d.c#L719-L744
    # 3 model parameters
    # rank = 3 => nu = size-3
    @staticmethod
    def robust_trend2d(data: np.ndarray, rank: int) -> np.ndarray:
        """
        Perform robust linear regression to estimate the trend in 2D data.

        Parameters
        ----------
        data : numpy.ndarray
            Array containing the input data. The shape of the array should be (N, 3), where N is the number of data points.
            The first column represents the x-coordinates, the second column represents the y-coordinates (if rank is 3),
            and the third column represents the z-values.
        rank : int
            Number of model parameters to fit. Should be 1, 2, or 3. If rank is 1, the function fits a constant trend.
            If rank is 2, it fits a linear trend. If rank is 3, it fits a planar trend.

        Returns
        -------
        numpy.ndarray
            Array containing the estimated trend coefficients. The length of the array depends on the specified rank.
            For rank 1, the array will contain a single value (intercept).
            For rank 2, the array will contain two values (intercept and slope).
            For rank 3, the array will contain three values (intercept, slope_x, slope_y).

        Raises
        ------
        Exception
            If the specified rank is not 1, 2, or 3.

        Notes
        -----
        The function performs robust linear regression using the M-estimator technique. It iteratively fits a linear model
        and updates the weights based on the residuals until convergence. The weights are adjusted using Tukey's bisquare
        weights to downweight outliers.

        References
        ----------
        - Rousseeuw, P. J. (1984). Least median of squares regression. Journal of the American statistical Association, 79(388), 871-880.

        - Huber, P. J. (1973). Robust regression: asymptotics, conjectures and Monte Carlo. The Annals of Statistics, 1(5), 799-821.
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression
        # scale factor for normally distributed data is 1.4826
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html
        MAD_NORMALIZE = 1.4826
        # significance value
        sig_threshold = 0.51

        if rank not in [1,2,3]:
            raise Exception('Number of model parameters "rank" should be 1, 2, or 3')

        #see gmt_stat.c
        def gmtstat_f_q (chisq1, nu1, chisq2, nu2):
            import scipy.special as sc

            if chisq1 == 0.0:
                return 1
            if chisq2 == 0.0:
                return 0
            return sc.betainc(0.5*nu2, 0.5*nu1, chisq2/(chisq2+chisq1))

        if rank in [2,3]:
            x = data[:,0]
            x = np.interp(x, (x.min(), x.max()), (-1, +1))
        if rank == 3:
            y = data[:,1]
            y = np.interp(y, (y.min(), y.max()), (-1, +1))
        z = data[:,2]
        w = np.ones(z.shape)

        if rank == 1:
            xy = np.expand_dims(np.zeros(z.shape),1)
        elif rank == 2:
            xy = np.expand_dims(x,1)
        elif rank == 3:
            xy = np.stack([x,y]).transpose()

        # create linear regression object
        mlr = LinearRegression()

        chisqs = []
        coeffs = []
        while True:
            # fit linear regression
            mlr.fit(xy, z, sample_weight=w)

            r = np.abs(z - mlr.predict(xy))
            chisq = np.sum((r**2*w))/(z.size-3)    
            chisqs.append(chisq)
            k = 1.5 * MAD_NORMALIZE * np.median(r)
            w = np.where(r <= k, 1, (2*k/r) - (k * k/(r**2)))
            sig = 1 if len(chisqs)==1 else gmtstat_f_q(chisqs[-1], z.size-3, chisqs[-2], z.size-3)
            # Go back to previous model only if previous chisq < current chisq
            if len(chisqs)==1 or chisqs[-2] > chisqs[-1]:
                coeffs = [mlr.intercept_, *mlr.coef_]

            #print ('chisq', chisq, 'significant', sig)
            if sig < sig_threshold:
                break

        # get the slope and intercept of the line best fit
        return (coeffs[:rank])

    # fitoffset.csh 3 3 freq_xcorr.dat
    # PRM.fitoffset(3, 3, offset_dat)
    # PRM.fitoffset(3, 3, matrix_fromfile='raw/offset.dat')
    @staticmethod
    def fitoffset(rank_rng: int, rank_azi: int, matrix: np.ndarray|None=None, matrix_fromfile: str|None=None, SNR: int=20) -> "PRM":
        """
        Estimates range and azimuth offsets for InSAR (Interferometric Synthetic Aperture Radar) data.

        Parameters
        ----------
        rank_rng : int
            Number of parameters to fit in the range direction.
        rank_azi : int
            Number of parameters to fit in the azimuth direction.
        matrix : numpy.ndarray, optional
            Array of range and azimuth offset estimates. Default is None.
        matrix_fromfile : str, optional
            Path to a file containing range and azimuth offset estimates. Default is None.
        SNR : int, optional
            Signal-to-noise ratio cutoff. Data points with SNR below this threshold are discarded.
            Default is 20.

        Returns
        -------
        prm : PRM object
            An instance of the PRM class with the calculated parameters.

        Raises
        ------
        Exception
            If both 'matrix' and 'matrix_fromfile' arguments are provided or if neither is provided.
        Exception
            If there are not enough data points to estimate the parameters.

        Usage
        -----
        The function estimates range and azimuth offsets for InSAR data based on the provided input.
        It performs robust fitting to obtain range and azimuth coefficients, calculates scale coefficients,
        and determines the range and azimuth shifts. The resulting parameters are then stored in a PRM object.

        Example
        -------
        fitoffset(3, 3, matrix_fromfile='raw/offset.dat')
        """
        import numpy as np
        import math

        if (matrix is None and matrix_fromfile is None) or (matrix is not None and matrix_fromfile is not None):
            raise Exception('One and only one argument matrix or matrix_fromfile should be defined')
        if matrix_fromfile is not None:
            matrix = np.genfromtxt(matrix_fromfile)

        #  first extract the range and azimuth data
        rng = matrix[np.where(matrix[:,4]>SNR)][:,[0,2,1]]
        azi = matrix[np.where(matrix[:,4]>SNR)][:,[0,2,3]]

        # make sure there are enough points remaining
        if rng.shape[0] < 8:
            raise Exception(f'FAILED - not enough points to estimate parameters, try lower SNR ({rng.shape[0]} < 8)')

        rng_coef = PRM.robust_trend2d(rng, rank_rng)
        azi_coef = PRM.robust_trend2d(azi, rank_azi)

        # print MSE (optional)
        #rng_mse = PRM.robust_trend2d_mse(rng, rng_coef, rank_rng)
        #azi_mse = PRM.robust_trend2d_mse(azi, azi_coef, rank_azi)
        #print ('rng_mse_norm', rng_mse/len(rng), 'azi_mse_norm', azi_mse/len(azi))

        # range and azimuth data ranges
        scale_coef = [np.min(rng[:,0]), np.max(rng[:,0]), np.min(rng[:,1]), np.max(rng[:,1])]

        #print ('rng_coef', rng_coef)
        #print ('azi_coef', azi_coef)

        # now convert to range coefficients
        rshift = rng_coef[0] - rng_coef[1]*(scale_coef[1]+scale_coef[0])/(scale_coef[1]-scale_coef[0]) \
            - rng_coef[2]*(scale_coef[3]+scale_coef[2])/(scale_coef[3]-scale_coef[2])
        # now convert to azimuth coefficients
        ashift = azi_coef[0] - azi_coef[1]*(scale_coef[1]+scale_coef[0])/(scale_coef[1]-scale_coef[0]) \
            - azi_coef[2]*(scale_coef[3]+scale_coef[2])/(scale_coef[3]-scale_coef[2])
        #print ('rshift', rshift, 'ashift', ashift)

        # note: Python x % y expression and nympy results are different to C, use math function
        # use 'g' format for float values as in original GMTSAR codes to easy compare results
        prm = PRM().set(rshift     =int(rshift) if rshift>=0 else int(rshift)-1,
                        sub_int_r  =math.fmod(rshift, 1)  if rshift>=0 else math.fmod(rshift, 1) + 1,
                        stretch_r  =rng_coef[1]*2/(scale_coef[1]-scale_coef[0]),
                        a_stretch_r=rng_coef[2]*2/(scale_coef[3]-scale_coef[2]),
                        ashift     =int(ashift) if ashift>=0 else int(ashift)-1,
                        sub_int_a  =math.fmod(ashift, 1)  if ashift>=0 else math.fmod(ashift, 1) + 1,
                        stretch_a  =azi_coef[1]*2/(scale_coef[1]-scale_coef[0]),
                        a_stretch_a=azi_coef[2]*2/(scale_coef[3]-scale_coef[2]),
                       )

        return prm

    def bounds(self) -> tuple[int, int]:
        maxx, yvalid, num_patch = self.get('num_rng_bins', 'num_valid_az', 'num_patches')
        maxy = yvalid * num_patch
        return [maxy, maxx]
