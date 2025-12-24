import astropy.units as u
import numpy as np
from enterprise.signals import parameter
from enterprise.signals.gp_signals import BasisGP, get_timing_model_basis
from enterprise.signals.selections import Selection, no_selection
from enterprise.signals.utils import (
    createfourierdesignmatrix_dm,
    createfourierdesignmatrix_red,
    tm_prior,
)
from enterprise.signals.white_signals import WhiteNoise
from enterprise.signals.signal_base import function
from pint import DMconst, dmu


def WidebandTimingModel(name="linear_wideband_timing_model"):
    """Class factory for marginalized linear timing model signals."""

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
    name="",
):
    """Class factory for EFAC+EQUAD and DMEFAC+DMEQUAD measurement noise
    (with tempo/tempo2/pint parameter convention, variance = efac^2 (toaerr^2 + t2equad^2)).
    """

    varianceFunction = wideband_ndiag(
        efac=efac,
        log10_t2equad=log10_t2equad,
        dmefac=dmefac,
        log10_dmequad=log10_dmequad,
    )
    BaseClass = WhiteNoise(varianceFunction, selection=selection, name=name)

    class MeasurementNoise(BaseClass):
        signal_name = "wideband_measurement_noise"
        signal_id = (
            "wideband_measurement_noise_" + name
            if name
            else "wideband_measurement_noise"
        )

    return MeasurementNoise


@function
def wideband_ndiag(
    toaerrs, dmerrs, efac=1.0, log10_t2equad=-8, dmefac=1.0, log10_dmequad=-8
):
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
    Ft, Ffreqs = createfourierdesignmatrix_red(
        toas, nmodes, Tspan, logf, fmin, fmax, pshift, modes, pseed
    )
    Ft[Ft.shape[0]:, :] = 0
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
    DMconst_value = DMconst.to_value(u.s * u.MHz**2 / dmu)
    Ft, Ffreqs = createfourierdesignmatrix_dm(
        toas, freqs, nmodes, Tspan, pshift, fref, logf, fmin, fmax, modes
    )
    Ft[Ft.shape[0] :, :] *= (freqs**2 / DMconst_value)[:, None]
    return Ft, Ffreqs
