from enterprise.signals.gp_signals import BasisGP
from enterprise.signals.parameter import Parameter, Uniform
from enterprise.signals.utils import powerlaw
from enterprise_extensions.dropout import dropout_powerlaw

from .signals import (
    createfourierdesignmatrix_dm_wideband,
    createfourierdesignmatrix_red_wideband,
    createfourierdesignmatrix_sw_wideband,
)


def get_powerlaw_spectrum(
    log10_A: Parameter,
    gamma: Parameter,
    components: int,
    dropbin: bool,
    dropbin_min: int,
    k_threshold: float,
):
    """Helper function to create a powerlaw spectrum."""
    if dropbin:
        k_dropbin = Uniform(dropbin_min, components + 1)
        return dropout_powerlaw(
            log10_A=log10_A,
            gamma=gamma,
            k_drop=1,
            k_dropbin=k_dropbin,
            k_threshold=k_threshold,
        )
    else:
        return powerlaw(log10_A=log10_A, gamma=gamma)


def achromatic_red_noise_powerlaw_block(
    log10_A: Parameter = Uniform(-18, -12),
    gamma: Parameter = Uniform(1, 7),
    components: int = 30,
    dropbin: bool = False,
    dropbin_min: int = 10,
    k_threshold: float = 0.5,
    name: str = "powerlaw_achromatic_red_noise",
):
    """Signal block for achromatic red noise with a powerlaw spectrum.
    Corresponds to PLRedNoise in PINT."""
    spectrum = get_powerlaw_spectrum(
        log10_A, gamma, components, dropbin, dropbin_min, k_threshold
    )
    basis = createfourierdesignmatrix_red_wideband(nmodes=components)
    return BasisGP(spectrum, basis, name=name)


def dm_noise_powerlaw_block(
    log10_A: Parameter = Uniform(-18, -12),
    gamma: Parameter = Uniform(1, 7),
    components: int = 30,
    dropbin: bool = False,
    dropbin_min: int = 10,
    k_threshold: float = 0.5,
    name: str = "powerlaw_dm_noise",
):
    """Signal block for DM noise with a powerlaw spectrum.
    Corresponds to PLDMNoise in PINT."""
    spectrum = get_powerlaw_spectrum(
        log10_A, gamma, components, dropbin, dropbin_min, k_threshold
    )
    basis = createfourierdesignmatrix_dm_wideband(nmodes=components)
    return BasisGP(spectrum, basis, name=name)


def solar_wind_noise_powerlaw_block(
    log10_A: Parameter = Uniform(-10, -5),
    gamma: Parameter = Uniform(1, 7),
    components: int = 30,
    dropbin: bool = False,
    dropbin_min: int = 10,
    k_threshold: float = 0.5,
    name: str = "powerlaw_sw_noise",
):
    """Signal block for DM noise with a powerlaw spectrum.
    Corresponds to PLDMNoise in PINT."""
    spectrum = get_powerlaw_spectrum(
        log10_A, gamma, components, dropbin, dropbin_min, k_threshold
    )
    basis = createfourierdesignmatrix_sw_wideband(nmodes=components)
    return BasisGP(spectrum, basis, name=name)
