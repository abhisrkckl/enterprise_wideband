import astropy.units as u
import numpy as np
from enterprise.signals.deterministic_signals import Deterministic
from enterprise.signals.selections import Selection, no_selection
from enterprise.signals.signal_base import function
from pint import DMconst, dmu


@function
def make_wideband_dm_signal(freqs, base_delay):
    DMconst_value = DMconst.to_value(u.s * u.MHz**2 / dmu)
    return base_delay * (freqs[(len(freqs) // 2) :] ** 2 / DMconst_value)


def WidebandDeterministic(
    waveform: function, selection=Selection(no_selection), name="", dispersion=False
):

    BaseClass = Deterministic(waveform, selection=selection, name=name)

    class WidebandDeterministic(BaseClass):
        def get_delay(self, params):
            base_delay = super().get_delay(params)

            dm_signal = (
                np.zeros_like(base_delay)
                if not dispersion
                else make_wideband_dm_signal(base_delay)
            )

            return np.append(base_delay, dm_signal)

    return WidebandDeterministic
