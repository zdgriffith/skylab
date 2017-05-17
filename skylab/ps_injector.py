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

import healpy as hp

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

    rot = np.copy(ev)

    # Function call
    rot["ra"], rot_dec = rotate(ev["trueRa"], ev["trueDec"],
                                ra * np.ones(len(ev)), dec * np.ones(len(ev)),
                                ev["ra"], np.arcsin(ev["sinDec"]))

    if "dec" in names:
        rot["dec"] = rot_dec
    rot["sinDec"] = np.sin(rot_dec)

    # exp information to save
    exp = ["dec", "sinDec", "ra", "logE", "sigma"]

    # mc information to delete
    mc = []
    for name in names:
      if name not in exp: mc.append(name)

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
        if not np.fabs(val) < np.pi / 2.:
            logger.warn("Source declination {0:2e} not in pi range".format(
                            val))
            return
        if not (np.sin(val) > self.sinDec_range[0]
                and np.sin(val) < self.sinDec_range[1]):
            logger.error("Injection declination not in sinDec_range!")
        self._src_dec = float(val)

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

        min_sinDec = max(A, sinDec - self.sinDec_bandwidth)
        max_sinDec = min(B, sinDec + self.sinDec_bandwidth)

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
                        str(key), N, np.degrees(self.src_dec)))

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        print("Selected {0:d} events in total".format(len(self.mc_arr)))

        self._weights()

        return

    def flux2mu(self, flux):
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

class TemplateInjector(Injector):
    r"""Class to inject MC from a source template to an event sample.

    """
    _src_dec = np.nan
    _sinDec_bandwidth = np.sin(1/180.*np.pi)
    _sinDec_range = [-1., 1.]

    _E0 = 1.
    _e_range = [0., np.inf]

    _random = np.random.RandomState()
    _seed = None

    def __init__(self, template, sinDec_bins, coords, gamma, **kwargs):
        r"""Constructor. Initialize the Injector class with basic
        characteristics regarding a point source.

        Parameters
        -----------
        template : healpix map as numpy array (all the keywords...)
            Signal map *without* smoothing.

        sinDec_bins : numpy array
            Sin(declination) bins.

        coords : str
            Coordinate system of template map ('equatorial' or 'galactic').

        gamma : float
            Spectral index, positive values for falling spectra

        kwargs : dict
            Set parameters of class different to default

        """

        # source properties
        self.gamma = gamma
        self.template = template
        self.sinDec_bins = sinDec_bins
        self.coords = coords

        # force template into dict object
        if not isinstance(self.template, dict):
            self.template = {-1: self.template}
            self.all_enums = [-1]
        else:
            self.all_enums = range(len(sample_probs))
        enum0 = self.all_enums[0]

        # min and max dec of template
        nside  = hp.get_nside(self.template[enum0])
        npix   = hp.nside2npix(nside)

        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        # compute declination range of template
        min_dec =  1
        max_dec = -1
        for p in range(npix):

          # skip empty pixels
          if self.template[enum0][p] <= 0: continue

          # coordinates of pixel in equatorial map
          (th, ph) = hp.pix2ang(nside,p)

          # compute declination
          dec = np.pi/2 - th
          if min_dec > dec: min_dec = dec
          if max_dec < dec: max_dec = dec

        # add fudge factors
        min_dec -= np.arcsin(self._sinDec_bandwidth)
        if (min_dec < -np.pi/2):
          min_dec = -np.pi/2
        max_dec += np.arcsin(self._sinDec_bandwidth)
        if (max_dec >  np.pi/2):
          max_dec =  np.pi/2

        # restrict to range of sinDec bins
        if (np.sin(min_dec) < self.sinDec_bins[0]):
          min_dec = np.arcsin(self.sinDec_bins[0])
        if (np.sin(max_dec) > self.sinDec_bins[-1]):
          max_dec = np.arcsin(self.sinDec_bins[-1])

        self._sinDec_range = [np.sin(min_dec), np.sin(max_dec)]
        delta = self._sinDec_range[1] - self._sinDec_range[0]

        self._setup()

        if delta < self._sinDec_bandwidth:
          raise ValueError("Sin(dec) range too small. Must be at least %.2e" % self._sinDec_bandwidth)

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

        return

    @property
    def src_dec(self):
        return self._src_dec

    @src_dec.setter
    def src_dec(self, val):
        if not np.fabs(val) < np.pi / 2.:
            logger.warn("Source declination {0:2e} not in pi range".format(
                            val))
            return
        if not (np.sin(val) > self.sinDec_range[0]
                and np.sin(val) < self.sinDec_range[1]):
            logger.error("Injection declination not in sinDec_range!")
        self._src_dec = float(val)

        return

    def _setup(self):
        r"""Set solid angles and declination bands.

        """

        # solid angle of selected events
        self._min_dec = np.arcsin(self._sinDec_range[0])
        self._max_dec = np.arcsin(self._sinDec_range[1])
        self._omega = 2. * np.pi * (self._sinDec_range[1] - self._sinDec_range[0])

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

    def fill(self, mc, livetime):
        r"""Fill the Injector with MonteCarlo events selecting events around
        the source position(s).

        Parameters
        -----------
        mc : recarray, dict of recarrays with sample enum as key (MultiPointSourceLLH)
            Monte Carlo events
        livetime : float, dict of floats
            Livetime per sample

        """

        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise ValueError("mc and livetime not compatible")

        self.mc = dict()
        self.mc_arr = np.empty(0, dtype=[("idx", np.int), ("enum", np.int),
                                         ("trueE", np.float), ("ow", np.float),
                                         ("ids", np.int), ("sinDec", np.float), ("trueDec", np.float)])

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        for key, mc_i in mc.iteritems():

            # append sinDec if not present
            if not "sinDec" in mc_i.dtype.fields:
                mc_i = np.lib.recfunctions.append_fields(
                        mc_i, "sinDec", np.sin(mc_i["dec"]),
                        dtypes=np.float, usemask=False)

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
            mc_arr["sinDec"] = self.mc[key]["sinDec"]
            mc_arr["trueDec"] = self.mc[key]["trueDec"]
            mc_arr["ids"] = self.ids( self.mc[key]["sinDec"] )

            self.mc_arr = np.append(self.mc_arr, mc_arr)

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        print("Selected {0:d} events in total".format(len(self.mc_arr)))

        self._weights()

        return

    def flux2mu(self, flux):
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

    def ids(self, sinDecs):
      r"""Calculate sin(dec) bin ids for each event"

      Parameters
      -----------
      sinDecs : array
        list of sin(dec) values

      Returns
      --------
        array with index of sinDec bin containing the event
      """
      return np.digitize(sinDecs, self.sinDec_bins)-1

    def sample(self, sample_probs, mean_signal, poisson=True):
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

        while True:

            # Generate event numbers using Poisson events.
            if poisson:
                num = self.random.poisson(mean_signal)
            else:
                num = int(np.around(mean_signal))

            logger.info("Mean number of events {0:.1f}".format(mean_signal))
            logger.info("Generated number of events {0:d}".format(num))

            if num < 1:
                # No events will be sampled.
                yield num, None
                continue

            # empty array with same structure as mc_arr
            sam_idx = np.empty(num, dtype=self.mc_arr.dtype)

            # append ra & dec
            sam_idx = np.lib.recfunctions.append_fields(sam_idx,
                                                        names=["ra","dec"],
                                                        dtypes=[np.float, np.float],
                                                        data=[np.empty(num), np.empty(num)],
                                                        usemask=False)
            # get sample number for each event
            sam_idx['enum'] = self.random.choice(self.all_enums, size=num, p = sample_probs)

            # count number of events per sample
            num_per_sample = np.array([len(sam_idx['enum'][sam_idx['enum']==i]) for i in self.all_enums])

            # keys to save from randomly chosen MC events
            keys = ['idx', 'enum', 'ow', 'trueE', 'ids']

            # loop through each sample enum
            for enum in self.all_enums:

                # get number of events in this sample
                num = num_per_sample[enum]
                if num == 0: continue

                # select events for this enum
                mask = (sam_idx['enum'] == enum)
                ev   =  sam_idx[mask]

                # randomly grab pixels from template
                nside = hp.get_nside(self.template[enum])
                npix  = hp.nside2npix(nside)
                pix   = self.random.choice(npix, size=num, p=self.template[enum])

                # compute theta & phi in equatorial coordinates
                if self.coords == 'galactic':
                    theta_gal, phi_gal = hp.pix2ang(nside, pix)
                    theta_eq,  phi_eq  = hp.Rotator(coord=['G','C'])(theta_gal, phi_gal)
                else:
                    theta_eq,  phi_eq  = hp.pix2ang(nside, pix)

                ev['ra'    ] = phi_eq
                ev['dec'   ] = np.pi/2. - theta_eq
                ev['sinDec'] = np.sin(ev['dec'])
                ev['ids'   ] = self.ids(ev['sinDec'])

                # compute number of events per ids
                n_ids = np.array([len(ev['ids'][ev['ids']==i]) for i in range(self.sinDec_bins.size)])
               
                # loop through events in sin(dec) bins
                for i, n in enumerate(n_ids):

                  # skip empty sin(dec) bins
                  if n <= 0: continue

                  # pull randomly from MC events with same ids
                  ids_mask = np.equal(self.mc_arr['ids'], i)
                  prob     = self._norm_w[ids_mask]/np.sum(self._norm_w[ids_mask])
                  sam      = self.random.choice(self.mc_arr[ids_mask], size=n, p=prob)
                 
                  # save sam info into ev 
                  for key in keys:
                    ev[key][ev['ids'] == i] = sam[key]
                    
                # copy ev back into sam_idx
                sam_idx[mask] = ev

            # get sample enums that contain events
            enums = np.unique(sam_idx['enum'])

            # if only one sample, just return recarray
            if len(enums) == 1 and enums[0] < 0:
                sam_ev = np.copy(self.mc[enums[0]][sam_idx["idx"]])

                yield num, rotate_struct(sam_ev, sam_idx['ra'], sam_idx['dec'])
                continue

            sam_ev = dict()
            for enum in enums:
                mask = (sam_idx['enum'] == enum)
                idx = sam_idx[mask]['idx']
                sam_ev_i = np.copy(self.mc[enum][idx])
                sam_ev[enum] = rotate_struct(sam_ev_i, sam_idx[mask]['ra'], sam_idx[mask]['dec'])

            yield num, sam_ev
