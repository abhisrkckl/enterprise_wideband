from enterprise.signals.gp_signals import BasisGP
from enterprise.signals.parameter import Parameter, Uniform
from enterprise.signals.utils import powerlaw

from .signals import (
    createfourierdesignmatrix_dm_wideband,
    createfourierdesignmatrix_red_wideband,
)


def achromatic_red_noise_powerlaw_block(
    log10_A: Parameter = Uniform(-18, -12),
    gamma: Parameter = Uniform(1, 7),
    components: int = 30,
):
    """Signal block for achromatic red noise with a powerlaw spectrum.
    Corresponds to PLRedNoise in PINT."""
    spectrum = powerlaw(log10_A=log10_A, gamma=gamma)
    basis = createfourierdesignmatrix_red_wideband(nmodes=components)
    return BasisGP(spectrum, basis, name="powerlaw_achromatic_red_noise")


def dm_noise_powerlaw_block(
    log10_A: Parameter = Uniform(-18, -12),
    gamma: Parameter = Uniform(1, 7),
    components: int = 30,
):
    """Signal block for DM noise with a powerlaw spectrum.
    Corresponds to PLDMNoise in PINT."""
    spectrum = powerlaw(log10_A=log10_A, gamma=gamma)
    basis = createfourierdesignmatrix_dm_wideband(nmodes=components)
    return BasisGP(spectrum, basis, name="powerlaw_dm_noise")


def solar_wind_noise_powerlaw_block(
    log10_A: Parameter = Uniform(-18, -12),
    gamma: Parameter = Uniform(1, 7),
    components: int = 30,
):
    """Signal block for DM noise with a powerlaw spectrum.
    Corresponds to PLDMNoise in PINT."""
    spectrum = powerlaw(log10_A=log10_A, gamma=gamma)
    basis = createfourierdesignmatrix_dm_wideband(nmodes=components)
    return BasisGP(spectrum, basis, name="powerlaw_sw_noise")
