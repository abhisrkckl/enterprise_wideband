import numpy as np
from enterprise.pulsar import PintPulsar
from pint import dmu
from pint.models import TimingModel
from pint.residuals import WidebandTOAResiduals
from pint.toa import TOAs


class WidebandPulsar(PintPulsar):
    """Represents a wideband pulsar dataset read using PINT. The residuals, measurement
    uncertainties, and the design matrix also include elements corresponding to the wideband
    DM measurements."""

    def __init__(
        self,
        toas: TOAs,
        model: TimingModel,
        sort: bool = True,
        drop_pintpsr: bool = True,
        planets: bool = True,
    ):
        assert toas.is_wideband(), "The TOAs are not wideband!"

        super().__init__(
            toas, model, sort=sort, drop_pintpsr=drop_pintpsr, planets=planets
        )

        self._dms = toas.get_dms().to_value(dmu).astype(float)
        self._dm_errors = toas.get_dm_errors().to_value(dmu).astype(float)

        self._wideband_residuals = (
            WidebandTOAResiduals(toas, model).calc_wideband_resids().astype(float)
        )
        self._wideband_designmatrix = model.wideband_designmatrix(toas)[0].astype(float)

        self._wideband_isort = np.append(self._isort, self.isort + len(self._toas))

        self._wideband_toas = np.append(self._toas, self._toas)

        self._wideband_freqs = np.append(self._ssbfreqs, self._ssbfreqs)

    @property
    def toas(self):
        """The MJD corresponding to each TOA. There are two measurements per TOA,
        so this array contains two copies of the TOA epochs."""
        return self._wideband_toas[self._wideband_isort]

    @property
    def freqs(self):
        """The observing frequency corresponding to each TOA. There are to measurements
        per TOA, so this array contains two copies of the frequencies."""
        return self._wideband_freqs[self._wideband_isort]

    @property
    def residuals(self):
        """An array containing the TOA residuals (s) and the DM residuals (pc/cm3)."""
        return self._wideband_residuals[self._wideband_isort]

    @property
    def Mmat(self):
        """Design matrix containing the derivatives of the TOA and DM residuals w.r.t. the
        timing model parameters. Also includes derivatives w.r.t. DMJUMPs."""
        return self._wideband_designmatrix[self._wideband_isort, :]

    @property
    def dmerrs(self):
        """An array containing the DM measurement uncertainties (pc/cm3)."""
        return self._dm_errors
