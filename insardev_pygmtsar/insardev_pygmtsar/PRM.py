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
        Correction for the range shift of the re-aligned SLC images (fix_prm_params() in GMTSAR).
        The clock_start and clock_stop attributes are not shifted: the span comes from them unchanged.
        """
        from scipy import constants
        # constant from GMTSAR code
        #SOL = 299792456.0
    
        delr = constants.speed_of_light / self.get('rng_samp_rate') / 2
        #delr = SOL / self.get('rng_samp_rate') / 2
        near_range = self.get('near_range') + \
            (self.get('st_rng_bin') - self.get('chirp_ext') + self.get('rshift') + self.get('sub_int_r') - 1)* delr
    
        return self.set(near_range=near_range)

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
            - dcPolynomial: Doppler centroid coefficients [c0, c1, c2] at burst center time, the ones the IPF
              used (utils_s1.dc_polynomial()): geometryDcPolynomial for a record flagged
              dataDcRmsErrorAboveThreshold, dataDcPolynomial otherwise
            - dcT0: Reference slant range time for DC polynomial
            - dcAzimuthTime: Azimuth time for DC estimate record

        Notes
        -----
        The FM rate and DC polynomials are time-varying. This method selects the record
        closest to the burst center time, following the same approach as GMTSAR's
        make_s1a_tops.c dramp_dmod() function.

        The XML annotation file is derived from the input_file path by replacing the measurement file
        extension (.nc, or the legacy .tiff) with .xml.

        References
        ----------
        GMTSAR make_s1a_tops.c: https://github.com/gmtsar/gmtsar
        ESA Sentinel-1 TOPS SLC Deramping: https://sentinels.copernicus.eu/documents/247904/1653442/Sentinel-1-TOPS-SLC_Deramping
        """
        from datetime import datetime
        import os
        import xmltodict
        from .utils_s1 import _aztime_seconds, dc_polynomial

        # Derive XML path from input_file (replace the .nc or .tiff extension with .xml)
        input_file = self.get('input_file')
        xml_file = os.path.splitext(input_file.replace('measurement','annotation'))[0] + '.xml'

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

        # Find azimuthFmRate record closest to burst center
        fm_rate_list = product['generalAnnotation']['azimuthFmRateList']['azimuthFmRate']
        if not isinstance(fm_rate_list, list):
            fm_rate_list = [fm_rate_list]

        best_fm_idx = 0
        best_fm_diff = float('inf')
        for i, fm_rate in enumerate(fm_rate_list):
            fm_time_str = fm_rate['azimuthTime']
            fm_time = datetime.strptime(fm_time_str, '%Y-%m-%dT%H:%M:%S.%f')
            # seconds from 00:00 UTC of the burst day, the clock of t_brst
            fm_t = _aztime_seconds(fm_time, burst_azimuth_time.date())
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
            # seconds from 00:00 UTC of the burst day, the clock of t_brst
            dc_t = _aztime_seconds(dc_time, burst_azimuth_time.date())
            diff = abs(dc_t - t_brst)
            if diff < best_dc_diff:
                best_dc_diff = diff
                best_dc_idx = i

        dc_record = dc_list[best_dc_idx]
        dc_azimuth_time = dc_record['azimuthTime']

        # the DC polynomial the IPF used
        dc_poly, dc_t0 = dc_polynomial(dc_record, xml_file)

        return {
            'azimuthSteeringRate': azimuth_steering_rate,
            'azimuthFmRatePolynomial': fm_poly,
            'azimuthFmRateT0': fm_t0,
            'azimuthFmRateAzimuthTime': fm_azimuth_time,
            'dcPolynomial': dc_poly,
            'dcT0': dc_t0,
            'dcAzimuthTime': dc_azimuth_time,
            'linesPerBurst': lpb,
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
    def robust_trend2d(data: np.ndarray, rank: int, debug: bool = False) -> np.ndarray:
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
        # maximum iterations to prevent infinite loops
        max_iterations = 100

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
        iteration = 0
        while True:
            iteration += 1
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
                if debug:
                    print(f"    robust_trend2d: converged in {iteration} iterations")
                break
            if iteration >= max_iterations:
                if debug:
                    print(f"    robust_trend2d: max iterations ({max_iterations}) reached")
                break

        # get the slope and intercept of the line best fit
        return (coeffs[:rank])

    # fitoffset.csh 3 3 freq_xcorr.dat
    # PRM.fitoffset(3, 3, offset_dat)
    # PRM.fitoffset(3, 3, matrix_fromfile='raw/offset.dat')
    @staticmethod
    def fitoffset(rank_rng: int, rank_azi: int, matrix: np.ndarray|None=None, matrix_fromfile: str|None=None, SNR: int=20, debug: bool=False) -> "PRM":
        """
        Estimates range and azimuth offsets for InSAR (Interferometric Synthetic Aperture Radar) data.

        Parameters
        ----------
        rank_rng : int
            Number of parameters to fit in the range direction: 1 (a constant shift), 2 (plus the stretch in
            range) or 3 (plus the stretch in azimuth, bilinear). The terms a lower rank does not fit are 0.
        rank_azi : int
            Number of parameters to fit in the azimuth direction, as rank_rng.
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

        rng_coef = PRM.robust_trend2d(rng, rank_rng, debug=debug)
        azi_coef = PRM.robust_trend2d(azi, rank_azi, debug=debug)

        # print MSE (optional)
        #rng_mse = PRM.robust_trend2d_mse(rng, rng_coef, rank_rng)
        #azi_mse = PRM.robust_trend2d_mse(azi, azi_coef, rank_azi)
        #print ('rng_mse_norm', rng_mse/len(rng), 'azi_mse_norm', azi_mse/len(azi))

        # range and azimuth data ranges
        scale_coef = [np.min(rng[:,0]), np.max(rng[:,0]), np.min(rng[:,1]), np.max(rng[:,1])]

        #print ('rng_coef', rng_coef)
        #print ('azi_coef', azi_coef)

        # now convert to shift and stretch coefficients; a term the rank does not fit is 0 (skipped, so a
        # constant range or azimuth of the points cannot turn it into 0 * inf)
        def shift_of(coef):
            shift = coef[0]
            if len(coef) > 1:
                shift -= coef[1]*(scale_coef[1]+scale_coef[0])/(scale_coef[1]-scale_coef[0])
            if len(coef) > 2:
                shift -= coef[2]*(scale_coef[3]+scale_coef[2])/(scale_coef[3]-scale_coef[2])
            return shift
        def stretch_of(coef):
            return coef[1]*2/(scale_coef[1]-scale_coef[0]) if len(coef) > 1 else 0.0
        def a_stretch_of(coef):
            return coef[2]*2/(scale_coef[3]-scale_coef[2]) if len(coef) > 2 else 0.0
        # range coefficients
        rshift = shift_of(rng_coef)
        # azimuth coefficients
        ashift = shift_of(azi_coef)
        #print ('rshift', rshift, 'ashift', ashift)

        # note: Python x % y expression and nympy results are different to C, use math function
        # use 'g' format for float values as in original GMTSAR codes to easy compare results
        prm = PRM().set(rshift     =int(rshift) if rshift>=0 else int(rshift)-1,
                        sub_int_r  =math.fmod(rshift, 1)  if rshift>=0 else math.fmod(rshift, 1) + 1,
                        stretch_r  =stretch_of(rng_coef),
                        a_stretch_r=a_stretch_of(rng_coef),
                        ashift     =int(ashift) if ashift>=0 else int(ashift)-1,
                        sub_int_a  =math.fmod(ashift, 1)  if ashift>=0 else math.fmod(ashift, 1) + 1,
                        stretch_a  =stretch_of(azi_coef),
                        a_stretch_a=a_stretch_of(azi_coef),
                       )

        return prm

    def bounds(self) -> tuple[int, int]:
        maxx, yvalid, num_patch = self.get('num_rng_bins', 'num_valid_az', 'num_patches')
        maxy = yvalid * num_patch
        return [maxy, maxx]
