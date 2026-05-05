from io import StringIO

import numpy as np
import pytest
from enterprise.signals.parameter import Uniform
from enterprise.signals.selections import Selection, no_selection, by_backend
from enterprise.signals.signal_base import PTA
from pint.config import examplefile
from pint.models import get_model, get_model_and_toas
from pint.simulation import make_fake_toas_uniform
from pint import DMconst, dmu
from astropy import units as u

from enterprise_wideband.blocks import (
    achromatic_red_noise_powerlaw_block,
    dm_noise_powerlaw_block,
    solar_wind_noise_powerlaw_block,
)
from enterprise_wideband.pulsar import WidebandPulsar
from enterprise_wideband.signals import WidebandMeasurementNoise, WidebandTimingModel


@pytest.fixture(scope="module")
def psr():
    parfile = examplefile("test-wb-0.par")
    timfile = examplefile("test-wb-0.tim")
    m, t = get_model_and_toas(parfile, timfile, planets=True)
    return WidebandPulsar(t, m)


@pytest.fixture(scope="module")
def psr2():
    par = """
        PSR                            J1234+5678
        EPHEM                               DE440
        CLOCK                        TT(BIPM2019)
        UNITS                                 TDB
        DILATEFREQ                              N
        DMDATA                                  N
        NTOA                                    0
        RAJ                     12:34:56.78900000 1 0.00000000000000000000
        DECJ                    56:00:00.12300000 1 0.00000000000000000000
        PMRA                                1e-10 0 0.0
        PMDEC                               1e-10 0 0.0
        PX                                    0.0
        POSEPOCH           55000.0000000000000000
        F0                                  100.0 1 0.0
        F1                                 -1e-15 1 0.0
        PEPOCH             55000.0000000000000000
        TNDMAMP                             -13.5
        TNDMGAM                               2.5
        TNDMC                                  50
        TNREDAMP                            -13.0
        TNREDGAM                              3.0
        TNREDC                                 30
        PLANET_SHAPIRO                          Y
        DM                                   10.0 1 0.0
        DMEPOCH            55000.0000000000000000
        TZRMJD             55000.2000400083124987
        TZRSITE                               ssb
        TZRFRQ                                inf
    """
    m = get_model(StringIO(par))
    tsim1 = make_fake_toas_uniform(
        model=m,
        startMJD=54000,
        endMJD=56000,
        ntoas=5000,
        error=0.5 * u.us,
        freq=1400 * u.MHz,
        add_noise=True,
        add_correlated_noise=True,
        wideband=True,
        flags={"f": "foo"},
        subtract_mean=False,
    )
    tsim2 = make_fake_toas_uniform(
        model=m,
        startMJD=54000,
        endMJD=56000,
        ntoas=5000,
        error=0.5 * u.us,
        freq=1000 * u.MHz,
        add_noise=True,
        add_correlated_noise=True,
        wideband=True,
        flags={"f": "bar"},
        subtract_mean=False,
    )
    t = tsim1 + tsim2
    return WidebandPulsar(t, m)


DMconst_value = DMconst.to_value(u.s * u.MHz**2 / dmu)


def test_pulsar(psr: WidebandPulsar):
    assert len(psr.toaerrs) == len(psr.dmerrs)
    assert psr.Mmat.shape[0] == len(psr.toaerrs) * 2
    assert len(psr.toaerrs) * 2 == len(psr.residuals)
    assert len(psr.freqs) == len(psr.residuals)
    assert psr.sunssb.shape[0] == len(psr.residuals)
    assert psr.planetssb.shape[0] == len(psr.residuals)


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
    assert np.allclose(
        basis[ntoas:, :] * DMconst_value / psr.freqs[ntoas:, None] ** 2,
        basis[:ntoas, :],
    )
    assert len(dmn_sig.params) == 2


def test_sw_noise(psr: WidebandPulsar):
    sw = solar_wind_noise_powerlaw_block()
    sw_sig = sw(psr)
    basis = sw_sig.get_basis()
    ntoas = len(psr.toas) // 2
    assert basis.shape[0] == ntoas * 2
    assert not np.all(basis[:ntoas, :] == 0)
    assert not np.all(basis[ntoas:, :] == 0)
    assert np.allclose(
        basis[ntoas:, :] * DMconst_value / psr.freqs[ntoas:, None] ** 2,
        basis[:ntoas, :],
    )
    assert len(sw_sig.params) == 2


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
    swn = solar_wind_noise_powerlaw_block()

    model = tm + wn + arn + dmn + swn

    pta = PTA([model(psr)])
    assert len(pta.param_names) == 10

    x0 = np.array([p.sample() for p in pta.params])
    x0_dict = pta.map_params(x0)

    n = pta.get_residuals()[0].size
    p = pta.get_phiinv(x0_dict)[0].size
    assert pta.get_ndiag(x0_dict)[0].size == n
    assert pta.get_basis(x0_dict)[0].shape == (n, p)

    assert np.isfinite(pta.get_lnprior(x0))
    assert np.isfinite(pta.get_lnlikelihood(x0))


def test_dropout_arn(psr: WidebandPulsar):
    arn = achromatic_red_noise_powerlaw_block(dropbin=True)
    arn_sig = arn(psr)
    basis = arn_sig.get_basis()
    ntoas = len(psr.toas) // 2
    assert basis.shape[0] == ntoas * 2
    assert not np.all(basis[:ntoas, :] == 0)
    assert np.all(basis[ntoas:, :] == 0)
    assert len(arn_sig.params) == 3


def test_dropout_dmn(psr: WidebandPulsar):
    dmn = dm_noise_powerlaw_block(dropbin=True)
    dmn_sig = dmn(psr)
    basis = dmn_sig.get_basis()
    ntoas = len(psr.toas) // 2
    assert basis.shape[0] == ntoas * 2
    assert not np.all(basis[:ntoas, :] == 0)
    assert not np.all(basis[ntoas:, :] == 0)
    assert np.allclose(
        basis[ntoas:, :] * DMconst_value / psr.freqs[ntoas:, None] ** 2,
        basis[:ntoas, :],
    )
    assert len(dmn_sig.params) == 3


def test_dropout_swn(psr: WidebandPulsar):
    sw = solar_wind_noise_powerlaw_block(dropbin=True)
    sw_sig = sw(psr)
    basis = sw_sig.get_basis()
    ntoas = len(psr.toas) // 2
    assert basis.shape[0] == ntoas * 2
    assert not np.all(basis[:ntoas, :] == 0)
    assert not np.all(basis[ntoas:, :] == 0)
    assert np.allclose(
        basis[ntoas:, :] * DMconst_value / psr.freqs[ntoas:, None] ** 2,
        basis[:ntoas, :],
    )
    assert len(sw_sig.params) == 3


def test_corrnoise_spna_dropout(psr: WidebandPulsar):
    tm = WidebandTimingModel()
    wn = WidebandMeasurementNoise(
        efac=Uniform(0.1, 2.5),
        log10_t2equad=Uniform(-8, -4),
        dmefac=Uniform(0.5, 1.5),
        log10_dmequad=Uniform(-8, -3),
        selection=Selection(no_selection),
        name="white_noise",
    )
    arn = achromatic_red_noise_powerlaw_block(dropbin=True)
    dmn = dm_noise_powerlaw_block(dropbin=True)
    swn = solar_wind_noise_powerlaw_block(dropbin=True)

    model = tm + wn + arn + dmn + swn

    pta = PTA([model(psr)])
    assert len(pta.param_names) == 13

    x0 = np.array([p.sample() for p in pta.params])
    x0_dict = pta.map_params(x0)

    n = pta.get_residuals()[0].size
    p = pta.get_phiinv(x0_dict)[0].size
    assert pta.get_ndiag(x0_dict)[0].size == n
    assert pta.get_basis(x0_dict)[0].shape == (n, p)

    assert np.isfinite(pta.get_lnprior(x0))
    assert np.isfinite(pta.get_lnlikelihood(x0))


def test_backend_selection(psr2: WidebandPulsar):
    assert len(psr2.backend_flags) == len(psr2.toas)

    tm = WidebandTimingModel()
    wn = WidebandMeasurementNoise(
        efac=Uniform(0.1, 2.5),
        log10_t2equad=Uniform(-8, -4),
        dmefac=Uniform(0.5, 1.5),
        log10_dmequad=Uniform(-8, -3),
        selection=Selection(by_backend),
    )
    arn = achromatic_red_noise_powerlaw_block()
    dmn = dm_noise_powerlaw_block()

    model = tm + wn + arn + dmn

    pta = PTA([model(psr2)])

    assert len(pta.param_names) == 12

    x0 = np.array([p.sample() for p in pta.params])
    assert np.isfinite(pta.get_lnlikelihood(x0))
