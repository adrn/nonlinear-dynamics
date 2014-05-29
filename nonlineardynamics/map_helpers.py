# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import base64
import glob
import os, sys

# Third-party
import h5py
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
from streamteam.dynamics import lyapunov_spectrum
from streamteam.integrate import DOPRI853Integrator

__all__ = ['LyapunovMap']

class LyapunovMap(object):

    def __init__(self, name, func, func_args=tuple(),
                 lyapunov_kwargs=dict(), Integrator=DOPRI853Integrator,
                 overwrite=False, prefix=""):
        """ Compute Lyapunov exponents for a grid of parameter values. This
            object provides a map()-like interface so the computation can
            be distributed via multiprocessing or MPI.

            Parameters
            ----------
            name : str
            func : callable
            func_args : sequence
                Extra arguments passed to the equations of motion function.
                These get appended to parameter arguments passed later.
            lyapunov_kwargs : keyword arguments
                Other arguments passed to `lyapunov()`. Things like the
                number of steps, timestep, etc.
            Integrator : streamteam.Integrator (optional)
            overwrite : bool (optional)
                Overwrite cached data files.
            prefix : str (optional)
                Prefix to the path.
        """

        # path to save data and plots
        self.output_path = os.path.join(prefix, "{}".format(name))
        self.cache_path = os.path.join(self.output_path, "cache")
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        self.F = func
        self._F_args = tuple(func_args)

        # class to use for integration
        self.Integrator = Integrator
        self.lyapunov_kwargs = lyapunov_kwargs

        self.overwrite = bool(overwrite)

        # initial conditions, potential parameters
        self.w0 = None
        self.potential_pars = None

    def _map_helper(self, w0, potential_pars, filename=None):
        """ """

        if filename is None:
            hashstr = "".join((str(w0) + str(potential_pars)).split())
            hashed = base64.b64encode(hashstr)
            fn = os.path.join(self.cache_path, "{}.hdf5".format(hashed))
        else:
            fn = os.path.join(self.cache_path, filename)

        if os.path.exists(fn) and self.overwrite:
            os.remove(fn)

        if not os.path.exists(fn):
            args = tuple(potential_pars) + self._F_args
            integrator = self.Integrator(self.F, func_args=args)
            LE,t,w = lyapunov_spectrum(w0, integrator, **self.lyapunov_kwargs)
            # LE,t,w = lyapunov_max(w0, integrator, **self.lyapunov_kwargs) #HACK

            with h5py.File(fn, "w") as f:
                f["lambda_k"] = LE
                f["t"] = t
                f["w"] = w
                f["potential_pars"] = potential_pars

    def __call__(self, arg):

        if self.w0 is None and self.potential_pars is not None:
            # assume arg is (index, w0)
            index,w0 = arg
            logger.debug("Index: {}".format(index))
            logger.debug("ICs: {}".format(w0))
            return self._map_helper(w0, self.potential_pars,
                                    filename="{}.hdf5".format(index))

        elif self.potential_pars is None and self.w0 is not None:
            # assume arg is (index, potential_pars)
            index,potential_pars = arg
            logger.debug("Index: {}".format(index))
            logger.debug("Pars: {}".format(potential_pars))
            return self._map_helper(self.w0, potential_pars,
                                    filename="{}.hdf5".format(index))

        else:
            raise ValueError("Must set either initial conditions or "
                             "potential parameters.")

    def iterate_cache(self):
        for fn in glob.glob(os.path.join(self.cache_path,"*.hdf5")):
            with h5py.File(fn, "r") as f:
                LE = np.array(f["lambda_k"].value)
                t = np.array(f["t"].value)
                w = np.array(f["w"].value)
                ppars = f["potential_pars"].value

            yield LE,t,w,ppars,fn

    def slope_diff(self, t, lambda_k):
        max_ix = lambda_k.sum(axis=0).argmax()
        lyap = lambda_k[:,max_ix]

        # fit line to first third and second 2/3, compare slopes
        logt = np.log10(t[1:])
        mid = 10**(2*(logt.max() + logt.min()) / 3.)
        ix = np.abs(t[1:] - mid).argmin() + 1

        # first third
        ms = []
        for this_t, this_lyap in [(t[1:ix], lyap[1:ix]), (t[ix:], lyap[ix:])]:
            # fit a line
            x = np.log10(this_t)
            y = np.log10(this_lyap)
            A = np.vstack([x, np.ones(len(x))]).T
            m,b = np.linalg.lstsq(A, y)[0]
            ms.append(m)

        return ms[1] - ms[0]

    def is_chaotic(self, t, lambda_k):
        dm = self.slope_diff(t, lambda_k)
        return False