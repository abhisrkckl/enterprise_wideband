import numpy as np
import pytest
from enterprise.signals.parameter import Uniform
from enterprise.signals.selections import Selection, no_selection
from enterprise.signals.signal_base import PTA
from pint.config import examplefile
from pint.models import get_model_and_toas

from enterprise_wideband.blocks import (
    achromatic_red_noise_powerlaw_block,
    dm_noise_powerlaw_block,
)
from enterprise_wideband.pulsar import WidebandPulsar
from enterprise_wideband.signals import WidebandMeasurementNoise, WidebandTimingModel


@pytest.fixture(scope="module")
def psr():
    parfile = examplefile("test-wb-0.par")
    timfile = examplefile("test-wb-0.tim")
    m, t = get_model_and_toas(parfile, timfile, planets=True)
    return WidebandPulsar(t, m)


def test_pulsar(psr: WidebandPulsar):
    assert len(psr.toaerrs) == len(psr.dmerrs)
    assert psr.Mmat.shape[0] == len(psr.toaerrs) * 2
    assert len(psr.toaerrs) * 2 == len(psr.residuals)


def test_timing_model(psr: WidebandPulsar):
    tm = WidebandTimingModel()
    tm_sig = tm(psr)
    assert tm_sig.name == f"{psr.name}_linear_wideband_timing_model"
    assert len(tm_sig.params) == 0
    assert tm_sig.get_basis().shape[0] == len(psr.toas)
    assert tm_sig.get_basis().shape[1] == len(tm_sig.get_phiinv([]))


def test_white_noise(psr: WidebandPulsar):
    wn = WidebandMeasurementNoise(
        log10_t2equad=Uniform(-8, -5),
        log10_dmequad=Uniform(-6, -3),
    )
    wn_sig = wn(psr)
    assert wn_sig.name == f"{psr.name}_wideband_white_noise"
    assert len(wn_sig.params) == 4
    params = {
        f"{psr.name}_efac": 1.0,
        f"{psr.name}_log10_t2equad": -7.0,
        f"{psr.name}_dmefac": 1.1,
        f"{psr.name}_log10_dmequad": -4.5,
    }
    assert len(wn_sig.get_ndiag(params)) == len(psr.toas)


def test_achromatic_red_noise(psr: WidebandPulsar):
    arn = achromatic_red_noise_powerlaw_block()
    arn_sig = arn(psr)
    basis = arn_sig.get_basis()
    ntoas = len(psr.toas) // 2
    assert basis.shape[0] == ntoas * 2
    assert not np.all(basis[:ntoas, :] == 0)
    assert np.all(basis[ntoas:, :] == 0)
    assert len(arn_sig.params) == 2


def test_dm_noise(psr: WidebandPulsar):
    dmn = dm_noise_powerlaw_block()
    dmn_sig = dmn(psr)
    basis = dmn_sig.get_basis()
    ntoas = len(psr.toas) // 2
    assert basis.shape[0] == ntoas * 2
    assert not np.all(basis[:ntoas, :] == 0)
    assert not np.all(basis[ntoas:, :] == 0)
    assert len(dmn_sig.params) == 2


def test_simple_spna(psr: WidebandPulsar):
    tm = WidebandTimingModel()
    wn = WidebandMeasurementNoise(
        efac=Uniform(0.1, 2.5),
        log10_t2equad=Uniform(-8, -4),
        dmefac=Uniform(0.5, 1.5),
        log10_dmequad=Uniform(-8, -3),
        selection=Selection(no_selection),
        name="white_noise",
    )

    model = tm + wn

    pta = PTA([model(psr)])
    assert len(pta.param_names) == 4

    x0 = np.array([p.sample() for p in pta.params])
    x0_dict = pta.map_params(x0)

    n = pta.get_residuals()[0].size
    p = pta.get_phiinv(x0_dict)[0].size
    assert pta.get_ndiag(x0_dict)[0].size == n
    assert pta.get_basis(x0_dict)[0].shape == (n, p)

    assert np.isfinite(pta.get_lnprior(x0))
    assert np.isfinite(pta.get_lnlikelihood(x0))


def test_corrnoise_spna(psr: WidebandPulsar):
    tm = WidebandTimingModel()
    wn = WidebandMeasurementNoise(
        efac=Uniform(0.1, 2.5),
        log10_t2equad=Uniform(-8, -4),
        dmefac=Uniform(0.5, 1.5),
        log10_dmequad=Uniform(-8, -3),
        selection=Selection(no_selection),
        name="white_noise",
    )
    arn = achromatic_red_noise_powerlaw_block()
    dmn = dm_noise_powerlaw_block()

    model = tm + wn + arn + dmn

    pta = PTA([model(psr)])
    assert len(pta.param_names) == 8

    x0 = np.array([p.sample() for p in pta.params])
    x0_dict = pta.map_params(x0)

    n = pta.get_residuals()[0].size
    p = pta.get_phiinv(x0_dict)[0].size
    assert pta.get_ndiag(x0_dict)[0].size == n
    assert pta.get_basis(x0_dict)[0].shape == (n, p)

    assert np.isfinite(pta.get_lnprior(x0))
    assert np.isfinite(pta.get_lnlikelihood(x0))
