import pytest
from enterprise.signals.parameter import Uniform
from enterprise_wideband.pulsar import WidebandPulsar
from enterprise_wideband.signals import WidebandMeasurementNoise, WidebandTimingModel
from pint.config import examplefile
from pint.models import get_model_and_toas


@pytest.fixture(scope="module")
def psr():
    parfile = examplefile("test-wb-0.par")
    timfile = examplefile("test-wb-0.tim")
    m, t = get_model_and_toas(parfile, timfile, planets=True)
    return WidebandPulsar(t, m)


def test_pulsar(psr: WidebandPulsar):
    assert len(psr.toaerrs) == len(psr.dmerrs)
    assert psr.Mmat.shape[0] == len(psr.toaerrs) * 2
    assert len(psr.toaerrs) * 2 == len(psr.wideband_residuals)


def test_timing_model(psr: WidebandPulsar):
    tm = WidebandTimingModel()
    tm_sig = tm(psr)
    assert tm_sig.name == f"{psr.name}_linear_wideband_timing_model"
    assert len(tm_sig.params) == 0
    assert tm_sig.get_basis().shape[0] == len(psr.toas)
    assert tm_sig.get_basis().shape[1] == len(tm_sig.get_phiinv([]))


def test_white_noise(psr: WidebandPulsar):
    wn = WidebandMeasurementNoise(
        log10_t2equad=Uniform(-8, -5), log10_dmequad=Uniform(-6, -3)
    )
    wn_sig = wn(psr)
    assert wn_sig.name == f"{psr.name}_wideband_measurement_noise"
    assert len(wn_sig.params) == 4
    params = {
        f"{psr.name}_efac": 1.0,
        f"{psr.name}_log10_t2equad": -7.0,
        f"{psr.name}_dmefac": 1.1,
        f"{psr.name}_log10_dmequad": -4.5,
    }
    assert len(wn_sig.get_ndiag(params)) == len(psr.toas)
