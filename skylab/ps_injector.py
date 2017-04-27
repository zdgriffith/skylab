# -*-coding:utf8-*-

from __future__ import print_function

"""
This file is part of SkyLab

Skylab is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

ps_injector
===========

Point Source Injection classes. The interface with the core
PointSourceLikelihood - Class requires the methods

    fill - Filling the class with Monte Carlo events

    sample - get a weighted sample with mean number `mu`

    flux2mu - convert from a flux to mean number of expected events

    mu2flux - convert from a mean number of expected events to a flux

"""

# python packages
import logging

# scipy-project imports
import numpy as np
from numpy.lib.recfunctions import drop_fields
import scipy.interpolate


# local package imports
from . import set_pars
from .utils import rotate

# get module logger
def trace(self, message, *args, **kwargs):
    r""" Add trace to logger with output level beyond debug

    """
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)

logging.addLevelName(5, "TRACE")
logging.Logger.trace = trace

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

_deg = 4
_ext = 3

def rotate_struct(ev, ra, dec):
    r"""Wrapper around the rotate-method in skylab.utils for structured
    arrays.

    Parameters
    ----------
    ev : structured array
        Event information with ra, sinDec, plus true information

    ra, dec : float
        Coordinates to rotate the true direction onto

    Returns
    --------
    ev : structured array
        Array with rotated value, true information is deleted

    """
    names = ev.dtype.names

    if not "sinDec" in names:
      ev = np.lib.recfunctions.append_fields(
               ev, "sinDec", np.sin(ev["dec"]),
               dtypes=np.float, usemask=False)

    rot = np.copy(ev)

    # Function call
    rot["ra"], rot_dec = rotate(ev["trueRa"], ev["trueDec"],
                                ra * np.ones(len(ev)), dec * np.ones(len(ev)),
                                ev["ra"], np.arcsin(ev["sinDec"]))

    if "dec" in names:
        rot["dec"] = rot_dec
    rot["sinDec"] = np.sin(rot_dec)

    # "delete" Monte Carlo information from sampled events
    mc = ["trueRa", "trueE", "ow"]
    if "trueDec"    in names: mc.append("trueDec")
    if "trueSinDec" in names: mc.append("trueSinDec")

    # "delete" extra conv, astro, and prompt weights
    # from diffuse muon neutrino samples made by Rene Reimann
    if "conv"       in names: mc.append("conv")
    if "astro"      in names: mc.append("astro")
    if "prompt"     in names: mc.append("prompt")

    return drop_fields(rot, mc)


class Injector(object):
    r"""Base class for Signal Injectors defining the essential classes needed
    for the LLH evaluation.

    """

    def __init__(self, *args, **kwargs):
        r"""Constructor: Define general point source features here...

        """
        self.__raise__()

    def __raise__(self):
        raise NotImplementedError("Implemented as abstract in {0:s}...".format(
                                    self.__repr__()))

    def fill(self, *args, **kwargs):
        r"""Filling the injector with the sample to draw from, work only on
        data samples known by the LLH class here.

        """
        self.__raise__()

    def flux2mu(self, *args, **kwargs):
        r"""Internal conversion from fluxes to event numbers.

        """
        self.__raise__()

    def mu2flux(self, *args, **kwargs):
        r"""Internal conversion from mean number of expected neutrinos to
        point source flux.

        """
        self.__raise__()

    def sample(self, *args, **kwargs):
        r"""Generator method that returns sampled events. Best would be an
        infinite loop.

        """
        self.__raise__()


class PointSourceInjector(Injector):
    r"""Class to inject a point source into an event sample.

    """
    _src_dec = np.nan
    _sinDec_bandwidth = np.sin(1/180.*np.pi)
    _sinDec_range = [-1., 1.]

    _E0 = 1.
    _e_range = [0., np.inf]

    _random = np.random.RandomState()
    _seed = None

    def __init__(self, gamma, **kwargs):
        r"""Constructor. Initialize the Injector class with basic
        characteristics regarding a point source.

        Parameters
        -----------
        gamma : float
            Spectral index, positive values for falling spectra

        kwargs : dict
            Set parameters of class different to default

        """

        # source properties
        self.gamma = gamma

        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        return

    def __str__(self):
        r"""String representation showing some more or less useful information
        regarding the Injector class.

        """
        sout = ("\n{0:s}\n"+
                67*"-"+"\n"+
                "\tSpectral index     : {1:6.2f}\n"+
                "\tSource declination : {2:5.1f} deg\n"
                "\tlog10 Energy range : {3:5.1f} to {4:5.1f}\n").format(
                         self.__repr__(),
                         self.gamma, np.degrees(self.src_dec),
                         *self.e_range)
        sout += 67*"-"

        return sout

    @property
    def sinDec_range(self):
        return self._sinDec_range

    @sinDec_range.setter
    def sinDec_range(self, val):
        if len(val) != 2:
            raise ValueError("SinDec range needs only upper and lower bound!")
        if val[0] < -1 or val[1] > 1:
            logger.warn("SinDec bounds out of [-1, 1], clip to that values")
            val[0] = max(val[0], -1)
            val[1] = min(val[1], 1)
        if np.diff(val) <= 0:
            raise ValueError("SinDec range has to be increasing")
        self._sinDec_range = [float(val[0]), float(val[1])]
        return

    @property
    def e_range(self):
        return self._e_range

    @e_range.setter
    def e_range(self, val):
        if len(val) != 2:
            raise ValueError("Energy range needs upper and lower bound!")
        if val[0] < 0. or val[1] < 0:
            logger.warn("Energy range has to be non-negative")
            val[0] = max(val[0], 0)
            val[1] = max(val[1], 0)
        if np.diff(val) <= 0:
            raise ValueError("Energy range has to be increasing")
        self._e_range = [float(val[0]), float(val[1])]
        return

    @property
    def E0(self):
        return self._E0

    @E0.setter
    def E0(self, value):
        self._E0 = float(value)

        return

    @property
    def random(self):
        return self._random

    @random.setter
    def random(self, value):
        self._random = value

        return

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, val):
        logger.info("Setting global seed to {0:d}".format(int(val)))
        self._seed = int(val)
        self.random = np.random.RandomState(self.seed)

        return

    @property
    def sinDec_bandwidth(self):
        return self._sinDec_bandwidth

    @sinDec_bandwidth.setter
    def sinDec_bandwidth(self, val):
        if val < 0. or val > 1:
            logger.warn("Sin Declination bandwidth {0:2e} not valid".format(
                            val))
            val = min(1., np.fabs(val))
        self._sinDec_bandwidth = float(val)

        self._setup()

        return

    @property
    def src_dec(self):
        return self._src_dec

    @src_dec.setter
    def src_dec(self, val):
        if not np.all(np.fabs(val) < np.pi / 2.):
            logger.warn("Source declination {0:2e} not in pi range".format(
                            val))
            return
        if not (np.all(np.sin(val) > self.sinDec_range[0])
                and np.all(np.sin(val) < self.sinDec_range[1])):
            logger.error("Injection declination not in sinDec_range!")

        self._src_dec = np.atleast_1d(val).astype(float)

        self._setup()

        return

    def _setup(self):
        r"""If one of *src_dec* or *dec_bandwidth* is changed or set, solid
        angles and declination bands have to be re-set.

        """

        A, B = self._sinDec_range

        m = (A - B + 2. * self.sinDec_bandwidth) / (A - B)
        b = self.sinDec_bandwidth * (A + B) / (B - A)

        sinDec = m * np.sin(self.src_dec) + b

        min_sinDec = np.maximum(A, sinDec - self.sinDec_bandwidth)
        max_sinDec = np.minimum(B, sinDec + self.sinDec_bandwidth)

        self._min_dec = np.arcsin(min_sinDec)
        self._max_dec = np.arcsin(max_sinDec)

        # solid angle of selected events
        self._omega = 2. * np.pi * (max_sinDec - min_sinDec)

        return

    def _weights(self):
        r"""Setup weights for simple power law model:

                 dN/dE = A (E / E0)^-gamma

            where A has units of events / (GeV cm^2 s). We treat
            the 'events' in the numerator as implicit and say the
            units are [GeV^-1 cm^-2 s^-1].

            The units of A balance with One Weight [GeV cm^2 sr] * Livetime [s] / Solid Angle [sr]
            to yield the number of events expected in the given livetime.
            We leave out A for now because we multiply by it later.
        """
        # energy scaled weights (everything but A)
        self.mc_arr["ow"] *= (self.mc_arr["trueE"] / self.E0)**(-self.gamma) / self._omega # [GeV cm^2 s]

        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float) # [GeV cm^2 s]

        # normalized weights for probability
        self._norm_w = self.mc_arr["ow"] / self._raw_flux

        # double-check if no weight is dominating the sample
        if self._norm_w.max() > 0.1:
            logger.warn("Warning: Maximal weight exceeds 10%: {0:7.2%}".format(
                            self._norm_w.max()))

        return

    def fill(self, src_dec, mc, livetime):
        r"""Fill the Injector with MonteCarlo events selecting events around
        the source position(s).

        Parameters
        -----------
        src_dec : float, array-like
            Source location(s)
        mc : recarray, dict of recarrays with sample enum as key (MultiPointSourceLLH)
            Monte Carlo events
        livetime : float, dict of floats
            Livetime per sample

        """

        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise ValueError("mc and livetime not compatible")

        self.src_dec = src_dec

        self.mc = dict()
        self.mc_arr = np.empty(0, dtype=[("idx", np.int), ("enum", np.int),
                                         ("trueE", np.float), ("ow", np.float)])

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        for key, mc_i in mc.iteritems():
            # get MC event's in the selected energy and sinDec range
            band_mask = ((np.sin(mc_i["trueDec"]) > np.sin(self._min_dec))
                         &(np.sin(mc_i["trueDec"]) < np.sin(self._max_dec)))
            band_mask &= ((mc_i["trueE"] > self.e_range[0])
                          &(mc_i["trueE"] < self.e_range[1]))

            if not np.any(band_mask):
                print("Sample {0:d}: No events were selected!".format(key))
                self.mc[key] = mc_i[band_mask]

                continue

            self.mc[key] = mc_i[band_mask]

            N = np.count_nonzero(band_mask)
            mc_arr = np.empty(N, dtype=self.mc_arr.dtype)
            mc_arr["idx"] = np.arange(N)
            mc_arr["enum"] = key * np.ones(N)
            mc_arr["ow"] = self.mc[key]["ow"] * livetime[key] * 86400.
            mc_arr["trueE"] = self.mc[key]["trueE"]

            self.mc_arr = np.append(self.mc_arr, mc_arr)

            print("Sample {0:s}: Selected {1:6d} events at {2:7.2f}deg".format(
                        str(key), N, float(np.degrees(self.src_dec))))

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        print("Selected {0:d} events in total".format(len(self.mc_arr)))

        self._weights()

        return

    def flux2mu(self, A):
        r"""Convert a flux normalization to mean number of expected events.

        The flux is calculated as follows:

        .. math::

            \frac{dN}{dE} = A \left( \frac{E}{E_0} \right)^{-\gamma}

        A has units of events / (GeV cm^2 s) but we treat it as
        [GeV^-1 cm^-2 s^-1] because the 'events' are implicit

        """

        return self._raw_flux * A # [events]

    def mu2flux(self, mu):
        r"""Calculate the corresponding flux normalization [GeV^-1 cm^-2 s^-1]
        for a given number of mean source events.

        """

        return mu / self._raw_flux # [GeV^-1 cm^-2 s^-1]

    def sample(self, src_ra, mean_signal, poisson=True):
        r""" Generator to get sampled events for a Point Source location.

        Parameters
        -----------
        mean_signal : float
            Mean number of signal events to sample

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            sampled_events for each loop iteration, either as simple array or
            as dictionary for each sample

        Optional Parameters
        --------------------
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_signal*

        """

        # generate event numbers using poissonian events
        while True:
            num = (self.random.poisson(mean_signal)
                        if poisson else int(np.around(mean_signal)))

            logger.debug(("Generated number of sources: {0:3d} "+
                          "of mean {1:5.1f} sources").format(num, mean_signal))

            # if no events should be sampled, return nothing
            if num < 1:
                yield num, None
                continue

            sam_idx = self.random.choice(self.mc_arr, size=num, p=self._norm_w)

            # get the events that were sampled
            enums = np.unique(sam_idx["enum"])

            if len(enums) == 1 and enums[0] < 0:
                # only one sample, just return recarray
                sam_ev = np.copy(self.mc[enums[0]][sam_idx["idx"]])

                yield num, rotate_struct(sam_ev, src_ra, self.src_dec)
                continue

            sam_ev = dict()
            for enum in enums:
                idx = sam_idx[sam_idx["enum"] == enum]["idx"]
                sam_ev_i = np.copy(self.mc[enum][idx])
                sam_ev[enum] = rotate_struct(sam_ev_i, src_ra, self.src_dec)

            yield num, sam_ev


class ModelInjector(PointSourceInjector):
    r"""PointSourceInjector that weights events according to a specific model
    flux.

    Fluxes are measured in percent of the input flux.

    """

    def __init__(self, logE, logFlux, *args, **kwargs):
        r"""Constructor, setting up the weighting function.

        Parameters
        -----------
        logE : array
            Flux Energy in units log(*self.GeV*)

        logFlux : array
            Flux in units log(*self.GeV* / cm^2 s), i.e. log(E^2 dPhi/dE)

        Other Parameters
        -----------------
        deg : int
            Degree of polynomial for flux parametrization

        args, kwargs
            Passed to PointSourceInjector

        """

        deg = kwargs.pop("deg", _deg)
        ext = kwargs.pop("ext", _ext)

        s = np.argsort(logE)
        logE = logE[s]
        logFlux = logFlux[s]
        diff = np.argwhere(np.diff(logE) > 0)
        logE = logE[diff]
        logFlux = logFlux[diff]

        self._spline = scipy.interpolate.InterpolatedUnivariateSpline(
                            logE, logFlux, k=deg)

        # use default energy range of the flux parametrization
        kwargs.setdefault("e_range", [10.**np.amin(logE), 10.**np.amax(logE)])

        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        return

    def _weights(self):
        r"""Calculate weights, according to given flux parametrization.

        """
        print("If you are here, you need to check this code. It is untested.")
        exit(0)
        trueLogGeV = np.log10(self.mc_arr["trueE"])

        logF = self._spline(trueLogGeV)
        flux = np.power(10., logF - 2. * trueLogGeV)

        # remove NaN's, etc.
        m = (flux > 0.) & np.isfinite(flux)
        self.mc_arr = self.mc_arr[m]

        # assign flux to OneWeight
        self.mc_arr["ow"] *= flux[m] / self._omega

        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)

        self._norm_w = self.mc_arr["ow"] / self._raw_flux

        return

    def flux2mu(self, flux):
        r"""Convert a flux to number of expected events.

        """

        return self._raw_flux * flux

    def mu2flux(self, mu):
        r"""Convert a mean number of expected events to a flux.

        """

        return float(mu) / self._raw_flux


class StackingPointSourceInjector(PointSourceInjector):
    r"""PointSourceInjector that injects events which are
        shared across multiple stacked sources.

    """

    def __init__(self, *args, **kwargs):

        #Same initialisation as in injector class
        super(StackingPointSourceInjector, self).__init__(*args, **kwargs)
        return


    def fill(self, src_dec, mc, livetime, **kwargs):
        r"""Fill the Injector with MonteCarlo events selecting events around
        the source positions.

        Parameters
        -----------
        src_dec: array-like
            Source locations
        mc : recarray, dict of recarrays with sample enum as key (StackingMultiPointSourceLLH)
            Monte Carlo events
        livetime : float, dict of floats
            Livetime per sample

        """

        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise ValueError("mc and livetime not compatible")

        self.src_dec = src_dec
        self.w_theo = kwargs.pop('w_theo',np.atleast_1d(np.ones_like(src_dec,dtype=float)))
        self.w_theo /= self.w_theo.sum()

        self.mc = dict()
        self.mc_arr = np.empty(0, dtype=[("idx", np.int),("src_idx", np.int),
                                         ("enum", np.int),("trueE", np.float),
                                         ("ow", np.float)])

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        for key, mc_i in mc.iteritems():
            # get MC event's in the selected energy and sinDec range around each src declination
            band_mask = ((np.sin(mc_i["trueDec"]) > np.sin(self._min_dec)[:, np.newaxis])
                         &(np.sin(mc_i["trueDec"]) < np.sin(self._max_dec)[:, np.newaxis]))

            band_mask &= ((mc_i["trueE"] > self.e_range[0])
                          &(mc_i["trueE"] < self.e_range[1]))

            if not np.any(band_mask):
                print("Sample {0:d}: No events were selected!".format(key))
                self.mc[key] = mc_i[band_mask.any(axis=0)]

                continue

            # all mc events that are at least in one declination band
            total_mask = band_mask.any(axis=0)
            N = np.count_nonzero(total_mask)

            self.mc[key] = mc_i[total_mask]

            #adjust band mask, count number of total events selected
            band_mask = (band_mask.T[total_mask]).T
            n = np.count_nonzero(band_mask)

            # save mc info (idx, src_idx, enum, weight) of all selected events
            mc_arr = np.empty(n, dtype=self.mc_arr.dtype)

            # unravel mask to stack source columns on top of one another
            # thereby making a 1D array of length len(self.src_dec) * n
            _m = band_mask.ravel()

            mc_arr["idx"] = np.tile(np.arange(N),len(self.src_dec))[_m]
            mc_arr['src_idx'] = np.repeat(np.arange(len(self.src_dec)),band_mask.sum(axis=1))
            mc_arr["enum"] = key * np.ones(len(mc_arr['idx']))
            mc_arr["ow"] = np.tile(self.mc[key]['ow'],len(self.src_dec))[_m] * livetime[key] * 86400.
            mc_arr["trueE"] = np.tile(self.mc[key]['trueE'],len(self.src_dec))[_m]

            self.mc_arr = np.append(self.mc_arr, mc_arr)

            print("Sample {0:s}: Selected {1:6d} events for {2:6d} sources.".format(
                                        str(key), n, len(self.src_dec)))

        # per event solid angle to account for events in different dec bands
        omega = (self._omega)[self.mc_arr['src_idx']]

        # per event source weights for each source
        w_theo = (self.w_theo)[self.mc_arr['src_idx']]

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        print("Selected {0:d} events in total".format(len(self.mc_arr)))

        self._weights(omega, w_theo)

        return

    def _weights(self, omega, w_theo):
        r"""Setup weights for simple power law model:

                 dN/dE = A (E / E0)^-gamma

            where A has units of events / (GeV cm^2 s). We treat
            the 'events' in the numerator as implicit and say the
            units are [GeV^-1 cm^-2 s^-1].

            The units of A balance with One Weight [GeV cm^2 sr] * Livetime [s] / Solid Angle [sr]
            to yield the number of events expected in the given livetime.
            We leave out A for now because we multiply by it later.

            NOTE: w_theo is a unitless scaling factor between the different
                  stacked sources. Two sources with equal weights would give [0.5, 0.5]
                  where each 0.5 is multiplied by every MC event in the
                  declination band containing the source location.
        """
        # energy scaled weights (everything but A)
        self.mc_arr["ow"] *= (self.mc_arr["trueE"] / self.E0)**(-self.gamma) / omega * w_theo # [GeV cm^2 s]

        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float) # [GeV cm^2 s]

        # normalized weights for probability
        self._norm_w = self.mc_arr["ow"] / self._raw_flux

        # double-check if no weight is dominating the sample
        if self._norm_w.max() > 0.1:
            logger.warn("Warning: Maximal weight exceeds 10%: {0:7.2%}".format(
                            self._norm_w.max()))

        return

    def sample(self, src_ra, mean_signal, poisson=True):
        r""" Generator to get sampled events for a Point Source location.

        Parameters
        -----------
        mean_signal : float
            Mean number of signal events to sample

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            sampled_events for each loop iteration, either as simple array or
            as dictionary for each sample

        Optional Parameters
        --------------------
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_signal*

        """

        # generate event numbers using poissonian events
        while True:

            num = (self.random.poisson(mean_signal)
                        if poisson else int(np.around(mean_signal)))

            logger.debug(("Generated number of sources: {0:3d} "+
                          "of mean {1:5.1f} sources").format(num, mean_signal))

            # if no events should be sampled, return nothing
            if num < 1:
                yield num, None
                continue

            sam_idx = self.random.choice(self.mc_arr, size=num, p=self._norm_w)

            # get the events that were sampled
            enums = np.unique(sam_idx["enum"])

            if len(enums) == 1 and enums[0] < 0:
                # only one sample, just return recarray
                sam_ev = np.copy(self.mc[enums[0]][sam_idx["idx"]])

                yield num, rotate_struct(sam_ev, src_ra[sam_idx['src_idx']], self.src_dec[sam_idx['src_idx']])
                continue

            sam_ev = dict()
            for enum in enums:
                idx = sam_idx[sam_idx["enum"] == enum]["idx"]
                sam_ev_i = np.copy(self.mc[enum][idx])
                src_ind = sam_idx[sam_idx["enum"] == enum]["src_idx"]
                sam_ev[enum] = rotate_struct(sam_ev_i, src_ra[src_ind], self.src_dec[src_ind])

            yield num, sam_ev

