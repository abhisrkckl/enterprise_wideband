from enterprise_wideband.pulsar import WidebandPulsar
from pint.config import examplefile
from pint.models import get_model_and_toas


def test_pulsar():
    parfile = examplefile("test-wb-0.par")
    timfile = examplefile("test-wb-0.tim")
    m, t = get_model_and_toas(parfile, timfile, planets=True)
    psr = WidebandPulsar(t, m)

    assert len(psr.toaerrs) == len(psr.dmerrs)
    assert psr.Mmat.shape[0] == len(psr.toaerrs) * 2
    assert len(psr.toaerrs) * 2 == len(psr.wideband_residuals)
