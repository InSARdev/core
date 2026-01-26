# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
"""
PRM methods for satellite geometry computations.
Pure Python implementations using functions from utils_s1.py and utils_satellite.py.
All methods use in-memory orbit data (self.orbit_df) - no file I/O.
"""


class PRM_gmtsar:
    import numpy as np

    def calc_dop_orb(self, earth_radius: float = 0, doppler_centroid: float = 0,
                     inplace: bool = False, debug: bool = False) -> "PRM":
        """
        Calculate the Doppler orbit parameters.

        Pure Python implementation using in-memory orbit data.

        Parameters
        ----------
        earth_radius : float, optional
            The Earth radius. If set to 0, the radius will be calculated. Default is 0.
        doppler_centroid : float, optional
            The Doppler centroid (currently not used, for API compatibility).
        inplace : bool, optional
            If True, set results in current PRM object. Default is False.
        debug : bool, optional
            If True, print debug information. Default is False.

        Returns
        -------
        PRM
            PRM object with calculated Doppler orbit parameters.
        """
        import time
        from .PRM import PRM
        from .utils_s1 import doppler_centroid as calc_doppler

        if self.orbit_df is None:
            raise ValueError("No orbit data attached to PRM. Use make_burst() to create PRM with orbit data.")

        start_time = time.perf_counter()

        # Get required PRM parameters
        clock_start = self.get('clock_start')
        prf = self.get('PRF')
        near_range = self.get('near_range')
        num_rng_bins = self.get('num_rng_bins')
        num_valid_az = self.get('num_valid_az')
        num_patches = self.get('num_patches')
        nrows = self.get('nrows')
        ra = self.get('equatorial_radius') if 'equatorial_radius' in self.df.index else 6378137.0
        rc = self.get('polar_radius') if 'polar_radius' in self.df.index else 6356752.31424518

        # Compute Doppler orbit parameters
        params = calc_doppler(
            orbit_df=self.orbit_df,
            clock_start=clock_start,
            prf=prf,
            near_range=near_range,
            num_rng_bins=num_rng_bins,
            num_valid_az=num_valid_az,
            num_patches=num_patches,
            nrows=nrows,
            ra=ra,
            rc=rc
        )

        elapsed = time.perf_counter() - start_time
        if debug:
            print(f'PROFILE: calc_dop_orb {elapsed:.3f}s')

        # If earth_radius was provided, use it
        if earth_radius != 0:
            params['earth_radius'] = earth_radius

        # Create PRM with computed parameters
        prm = PRM().set(
            earth_radius=params['earth_radius'],
            SC_height=params['SC_height'],
            SC_height_start=params['SC_height_start'],
            SC_height_end=params['SC_height_end'],
            SC_vel=params['SC_vel'],
            orbdir=params['orbdir']
        )

        if inplace:
            return self.set(prm)
        else:
            return prm

    def SAT_llt2rat(self, coords: np.ndarray, precise: int = 1,
                    debug: bool = False) -> np.ndarray:
        """
        Convert geographic coordinates (LLT) to radar coordinates (RAT).

        Pure Python implementation using in-memory orbit data.

        Parameters
        ----------
        coords : array_like
            LLT coordinates with shape (N, 3): [longitude, latitude, elevation].
        precise : int, optional
            Precision level (0=standard, 1=polynomial refinement). Default is 1.
        debug : bool, optional
            If True, print debug information. Default is False.

        Returns
        -------
        numpy.ndarray
            RAT coordinates with shape (N, 5):
            [range_pix, azimuth_pix, range_m, azimuth_time, elevation].
        """
        import numpy as np
        import time
        from .utils_s1 import satellite_llt2rat

        if self.orbit_df is None:
            raise ValueError("No orbit data attached to PRM. Use make_burst() to create PRM with orbit data.")

        llt_coords = np.atleast_2d(coords)
        n_coords = len(llt_coords)

        if debug:
            print(f'DEBUG: SAT_llt2rat n={n_coords}')

        start_time = time.perf_counter()

        # Get required PRM parameters - all must be present
        result = satellite_llt2rat(
            lon=llt_coords[:, 0],
            lat=llt_coords[:, 1],
            elevation=llt_coords[:, 2],
            orbit_df=self.orbit_df,
            clock_start=self.get('clock_start'),
            prf=self.get('PRF'),
            near_range=self.get('near_range'),
            rng_samp_rate=self.get('rng_samp_rate'),
            num_valid_az=self.get('num_valid_az'),
            num_patches=self.get('num_patches'),
            nrows=self.get('nrows'),
            earth_radius=self.get('earth_radius'),
            ra=self.get('equatorial_radius'),
            rc=self.get('polar_radius'),
            precise=precise,
            lookdir=self.get('lookdir'),
            fd1=self.get('fd1'),
            fdd1=self.get('fdd1'),
            wavelength=self.get('radar_wavelength'),
            vel=self.get('SC_vel'),
            num_rng_bins=self.get('num_rng_bins'),
            rshift=int(self.get('rshift')),
            ashift=int(self.get('ashift')),
            sub_int_r=self.get('sub_int_r'),
            sub_int_a=self.get('sub_int_a'),
            chirp_ext=int(self.get('chirp_ext'))
        )

        elapsed = time.perf_counter() - start_time
        if debug:
            print(f'PROFILE: SAT_llt2rat n={n_coords} {elapsed:.3f}s')

        return result if result.shape[0] > 1 else result.ravel()

    def SAT_look(self, coords: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Compute the satellite look vector.

        Pure Python implementation using in-memory orbit data.

        Parameters
        ----------
        coords : array_like
            LLT coordinates with shape (N, 3): [longitude, latitude, elevation].
        debug : bool, optional
            If True, print debug information. Default is False.

        Returns
        -------
        numpy.ndarray
            Look vectors with shape (N, 6):
            [longitude, latitude, elevation, look_E, look_N, look_U].
        """
        import numpy as np
        import time
        from .utils_s1 import satellite_look

        if self.orbit_df is None:
            raise ValueError("No orbit data attached to PRM. Use make_burst() to create PRM with orbit data.")

        llt_coords = np.atleast_2d(coords)
        n_coords = len(llt_coords)

        if debug:
            print(f'DEBUG: SAT_look n={n_coords}')

        start_time = time.perf_counter()

        # Compute look vectors
        look_E, look_N, look_U = satellite_look(
            lon=llt_coords[:, 0],
            lat=llt_coords[:, 1],
            elevation=llt_coords[:, 2],
            orbit_df=self.orbit_df,
            clock_start=self.get('clock_start'),
            prf=self.get('PRF'),
            num_valid_az=self.get('num_valid_az'),
            num_patches=self.get('num_patches'),
            nrows=self.get('nrows'),
            earth_radius=self.get('earth_radius'),
            ra=self.get('equatorial_radius') if 'equatorial_radius' in self.df.index else 6378137.0,
            rc=self.get('polar_radius') if 'polar_radius' in self.df.index else 6356752.31424518
        )

        elapsed = time.perf_counter() - start_time
        if debug:
            print(f'PROFILE: SAT_look n={n_coords} {elapsed:.3f}s')

        # Build result array: [lon, lat, elevation, look_E, look_N, look_U]
        result = np.column_stack([
            llt_coords[:, 0], llt_coords[:, 1], llt_coords[:, 2],
            look_E, look_N, look_U
        ])

        return result if result.shape[0] > 1 else result.ravel()

    def SAT_baseline(self, other: "PRM", debug: bool = False) -> "PRM":
        """
        Compute the satellite baseline between two acquisitions.

        Pure Python implementation using in-memory orbit data.
        Uses GMTSAR's exact algorithm for compatibility.

        Parameters
        ----------
        other : PRM
            PRM object for the secondary image.
        debug : bool, optional
            If True, print debug information. Default is False.

        Returns
        -------
        PRM
            PRM object containing baseline parameters:
            - B_parallel, B_perpendicular, baseline
            - baseline_start, baseline_center, baseline_end
            - alpha_start, alpha_center, alpha_end
            - B_offset_start, B_offset_center, B_offset_end
            - SC_height, SC_height_start, SC_height_end
        """
        import time
        from .PRM import PRM
        from .utils_satellite import satellite_baseline

        if not isinstance(other, PRM):
            raise ValueError('Argument "other" should be PRM class instance')

        if self.orbit_df is None:
            raise ValueError("No orbit data attached to reference PRM.")
        if other.orbit_df is None:
            raise ValueError("No orbit data attached to secondary PRM.")

        start_time = time.perf_counter()

        result = satellite_baseline(
            orbit_df1=self.orbit_df,
            orbit_df2=other.orbit_df,
            clock_start=self.get('clock_start'),
            prf=self.get('PRF'),
            num_valid_az=int(self.get('num_valid_az')),
            num_patches=int(self.get('num_patches')),
            nrows=int(self.get('nrows')),
            # Additional parameters for GMTSAR-style computation
            earth_radius=self.get('earth_radius') if 'earth_radius' in self.df.index else None,
            SC_height=self.get('SC_height') if 'SC_height' in self.df.index else None,
            near_range=self.get('near_range') if 'near_range' in self.df.index else None,
            num_rng_bins=int(self.get('num_rng_bins')) if 'num_rng_bins' in self.df.index else None,
            rng_samp_rate=self.get('rng_samp_rate') if 'rng_samp_rate' in self.df.index else None,
            # Repeat image parameters
            clock_start_rep=other.get('clock_start') if 'clock_start' in other.df.index else None,
            num_valid_az_rep=int(other.get('num_valid_az')) if 'num_valid_az' in other.df.index else None,
            num_patches_rep=int(other.get('num_patches')) if 'num_patches' in other.df.index else None,
            nrows_rep=int(other.get('nrows')) if 'nrows' in other.df.index else None,
            prf_rep=other.get('PRF') if 'PRF' in other.df.index else None
        )

        elapsed = time.perf_counter() - start_time
        if debug:
            print(f'PROFILE: SAT_baseline {elapsed:.3f}s')
            print(f"DEBUG: baseline={result['baseline']:.1f}m, "
                  f"Bpar={result['B_parallel']:.1f}m, "
                  f"Bperp={result['B_perpendicular']:.1f}m")

        # Return as PRM object for compatibility with prm.set()
        return PRM().set(**result)
