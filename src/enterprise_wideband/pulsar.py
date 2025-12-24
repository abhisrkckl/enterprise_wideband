import numpy as np
from enterprise.pulsar import PintPulsar
from pint import dmu
from pint.models import TimingModel
from pint.residuals import WidebandTOAResiduals
from pint.toa import TOAs


class WidebandPulsar(PintPulsar):
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

    @property
    def toas(self):
        return self._wideband_toas[self._wideband_isort]

    @property
    def wideband_residuals(self):
        """Return array of residuals in seconds."""
        return self._wideband_residuals[self._wideband_isort]

    @property
    def Mmat(self):
        """Return ntoa x npar design matrix."""
        return self._wideband_designmatrix[self._wideband_isort, :]

    @property
    def dmerrs(self):
        return self._dm_errors
