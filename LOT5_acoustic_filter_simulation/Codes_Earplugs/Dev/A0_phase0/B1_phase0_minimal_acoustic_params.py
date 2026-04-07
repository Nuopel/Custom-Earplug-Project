"""Phase 0 minimal example: direct AcousticParameters usage."""

from __future__ import annotations

import numpy as np

from toolkitsd.acoustmm import AcousticParameters


def main() -> None:
    freqs = np.array([100.0, 250.0, 500.0, 1000.0, 2000.0])
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)

    print("frequencies [Hz]:", params.frequencies)
    print("wavenumbers [rad/m]:", np.round(params.wavenumbers, 6))
    print("c0 [m/s]:", params.c0)
    print("rho0 [kg/m^3]:", params.rho0)
    print("z0 [Pa*s/m]:", params.z0)


if __name__ == "__main__":
    main()
