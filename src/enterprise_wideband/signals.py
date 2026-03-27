import astropy.units as u
import numpy as np
from enterprise.signals import parameter
from enterprise.signals.gp_signals import BasisGP, get_timing_model_basis
from enterprise.signals.selections import Selection, no_selection
from enterprise.signals.signal_base import function
from enterprise.signals.utils import (
    createfourierdesignmatrix_dm,
    createfourierdesignmatrix_red,
    tm_prior,
)
from enterprise.signals.white_signals import WhiteNoise
from enterprise_extensions.chromatic.solar_wind import (
    createfourierdesignmatrix_solar_dm,
)
from pint import DMconst, dmu


def WidebandTimingModel(name="linear_wideband_timing_model"):
    """Class factory for marginalized linear timing model for wideband data."""

    basis = get_timing_model_basis(use_svd=False, normed=True, idx_exclude=None)
    prior = tm_prior()

    BaseClass = BasisGP(prior, basis, coefficients=False, name=name)

    class TimingModel(BaseClass):
        signal_type = "basis"
        signal_name = "linear wideband timing model"
        signal_id = name

    return TimingModel


def WidebandMeasurementNoise(
    efac=parameter.Uniform(0.5, 1.5),
    log10_t2equad=parameter.Constant(-18),
    dmefac=parameter.Uniform(0.5, 1.5),
    log10_dmequad=parameter.Constant(-18),
    selection=Selection(no_selection),
    name="wideband_white_noise",
):
    """Class factory for wideband measurement noise. The TOA uncertainties are modified by EFACs and EQUADS,
    and the DM uncertainties are modified by DMEFACs and DMEQUADs. Follows the tempo2/pint convention,
    where variance = efac^2 (toaerr^2 + t2equad^2).
    """

    varianceFunction = wideband_ndiag(
        efac=efac,
        log10_t2equad=log10_t2equad,
        dmefac=dmefac,
        log10_dmequad=log10_dmequad,
    )
    BaseClass = WhiteNoise(varianceFunction, selection=selection, name=name)

    class MeasurementNoise(BaseClass):
        signal_name = "wideband_white_noise"
        signal_id = name

    return MeasurementNoise


@function
def wideband_ndiag(
    toaerrs, dmerrs, efac=1.0, log10_t2equad=-8, dmefac=1.0, log10_dmequad=-8
):
    """Create diagonal white noise covariance matrix for wideband data."""
    equad = 10**log10_t2equad
    dmequad = 10**log10_dmequad

    ntoa = len(toaerrs)

    Ndiag = np.empty(2 * ntoa)
    Ndiag[:ntoa] = efac**2 * (toaerrs**2 + equad**2)
    Ndiag[ntoa:] = dmefac**2 * (dmerrs**2 + dmequad**2)

    return Ndiag


@function
def createfourierdesignmatrix_red_wideband(
    toas,
    nmodes=30,
    Tspan=None,
    logf=False,
    fmin=None,
    fmax=None,
    pshift=False,
    modes=None,
    pseed=None,
):
    """Create achromatic red noise basis matrix for wideband data. The entries corresponding
    to the DM measurements are zero."""
    Ft, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes, Tspan, logf, fmin, fmax, pshift, modes, pseed
    )
    Ft[(len(toas) // 2) :, :] = 0
    return Ft, Ffreqs


@function
def createfourierdesignmatrix_dm_wideband(
    toas,
    freqs,
    nmodes=30,
    Tspan=None,
    pshift=False,
    fref=1400,
    logf=False,
    fmin=None,
    fmax=None,
    modes=None,
):
    """Create DM noise basis matrix for wideband data."""
    DMconst_value = DMconst.to_value(u.s * u.MHz**2 / dmu)
    Ft, Ffreqs = createfourierdesignmatrix_dm(
        toas, freqs, nmodes, Tspan, pshift, fref, logf, fmin, fmax, modes
    )
    Ft[(len(toas) // 2) :, :] *= (freqs[(len(toas) // 2) :] ** 2 / DMconst_value)[
        :, None
    ]
    return Ft, Ffreqs


@function
def createfourierdesignmatrix_sw_wideband(
    toas,
    freqs,
    planetssb,
    sunssb,
    pos_t,
    modes=None,
    nmodes=30,
    Tspan=None,
    logf=False,
    fmin=None,
    fmax=None,
):
    """Create Solar Wind basis matrix for wideband data."""
    DMconst_value = DMconst.to_value(u.s * u.MHz**2 / dmu)
    Ft, Ffreqs = createfourierdesignmatrix_solar_dm(
        toas, freqs, planetssb, sunssb, pos_t, modes, nmodes, Tspan, logf, fmin, fmax
    )
    Ft[(len(toas) // 2) :, :] *= (freqs[(len(toas) // 2) :] ** 2 / DMconst_value)[
        :, None
    ]
    return Ft, Ffreqs
