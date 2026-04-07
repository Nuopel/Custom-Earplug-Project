from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ACOUSTMM_ROOT = HERE.parents[2]
REFACTOR_ROOT = HERE.parents[3]
candidate_paths = [ACOUSTMM_ROOT / "src"]
candidate_paths.extend(sorted(REFACTOR_ROOT.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import AcousticElement, CylindricalDuct
from toolkitsd.porous import BasePorousMaterial, EquivalentFluidModel, MikiMaterial, MikiModel, build_porous_medium_props


def rigid_backed_surface_impedance_from_porous(
    material: BasePorousMaterial,
    frequencies_hz: np.ndarray,
    *,
    incidence_angle_deg: float = 90.0,
    model: EquivalentFluidModel | None = None,
) -> np.ndarray:
    """Return rigid-backed porous surface impedance for a locally reacting wall."""
    porous_props = build_porous_medium_props(material, frequencies_hz, model=model or MikiModel())
    response = porous_props.surface_response_on_rigid_backing(incidence_angle_deg=incidence_angle_deg)
    return np.asarray(response.surface_impedance, dtype=np.complex128).reshape(-1)


def calculate_kz_circular_lined_duct(
    k0: np.ndarray,
    *,
    c0: float,
    rho0: float,
    zs_wall: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Approximate axial wavenumber for a circular duct with locally reacting lining.

    Munjal, Section 6.4:
    - Eq. 6.62 gives the approximation of (kr,0 * r0)^2
    - Eq. 6.65 gives alpha0 = -Im(sqrt(k0^2 - kr^2))
    - the selected branch is the one that yields the lower attenuation
    """
    q = (k0 * radius) * (rho0 * c0) / zs_wall

    kr2_r2_plus = (96.0 + 36.0j * q + np.sqrt(9216.0 + 2304.0j * q - 912.0 * q**2)) / (12.0 + 1.0j * q)
    kr2_r2_minus = (96.0 + 36.0j * q - np.sqrt(9216.0 + 2304.0j * q - 912.0 * q**2)) / (12.0 + 1.0j * q)

    kr2_plus = kr2_r2_plus / radius**2
    kr2_minus = kr2_r2_minus / radius**2
    kz_plus = np.sqrt(k0**2 - kr2_plus + 0j)
    kz_minus = np.sqrt(k0**2 - kr2_minus + 0j)

    alpha_plus = -np.imag(kz_plus)
    alpha_minus = -np.imag(kz_minus)
    return np.where(alpha_plus <= alpha_minus, kz_plus, kz_minus)


class LinedCylindricalDuct(AcousticElement):
    """Minimal local prototype of a cylindrical lined duct."""

    def __init__(
        self,
        *,
        radius: float,
        length: float,
        frequencies_hz: np.ndarray,
        lining_material: BasePorousMaterial | None = None,
        zs_wall: np.ndarray | None = None,
        incidence_angle_deg: float = 90.0,
        porous_model: EquivalentFluidModel | None = None,
        c0: float = 343.0,
        rho0: float = 1.2,
    ) -> None:
        self.radius = float(radius)
        self.length = float(length)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.area = np.pi * self.radius**2
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64).ravel()
        self.omega = 2.0 * np.pi * self.frequencies_hz
        k0 = self.omega / self.c0

        if zs_wall is None:
            if lining_material is None:
                raise ValueError("Provide either zs_wall or lining_material.")
            zs_wall = rigid_backed_surface_impedance_from_porous(
                lining_material,
                self.frequencies_hz,
                incidence_angle_deg=incidence_angle_deg,
                model=porous_model,
            )
        self.zs_wall = np.asarray(zs_wall, dtype=np.complex128).ravel()
        self.kz = calculate_kz_circular_lined_duct(
            k0,
            c0=self.c0,
            rho0=self.rho0,
            zs_wall=self.zs_wall,
            radius=self.radius,
        )
        self.zc_lossless = self.rho0 * self.c0 / self.area
        self.zc_eff = self.rho0 * self.omega / (self.kz * self.area)

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if omega.shape != self.omega.shape or not np.allclose(omega, self.omega):
            raise ValueError("This prototype expects the same frequency grid used at initialization")

        kzL = self.kz * self.length
        ckzL = np.cos(kzL)
        skzL = np.sin(kzL)

        t = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        t[:, 0, 0] = ckzL
        t[:, 0, 1] = 1j * self.zc_eff * skzL
        t[:, 1, 0] = 1j * skzL / self.zc_eff
        t[:, 1, 1] = ckzL
        return t


def plot_tl_circular_comparison(freqs_hz: np.ndarray, tl_rigid_db: np.ndarray, tl_lined_db: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs_hz, tl_rigid_db, linewidth=2.0, label="Rigid cylindrical duct")
    plt.semilogx(freqs_hz, tl_lined_db, linewidth=2.0, label="Lined cylindrical duct")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Transmission Loss [dB]")
    plt.title("Minimal lined circular duct test")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="best")


if __name__ == "__main__":
    C0 = 343.0
    RHO0 = 1.2

    freqs_hz = np.logspace(np.log10(50.0), np.log10(5000.0), 300)
    omega = 2.0 * np.pi * freqs_hz

    radius_cyl = 0.01
    length_duct = 0.1

    rigid_cyl_duct = CylindricalDuct(radius=radius_cyl, length=length_duct, c0=C0, rho0=RHO0)

    lining_material = MikiMaterial(
        sigma=15000.0,
        thickness=0.005,
        rho0=RHO0,
        c0=C0,
        name="Miki lining",
    )
    circular_lined_duct = LinedCylindricalDuct(
        radius=radius_cyl,
        length=length_duct,
        frequencies_hz=freqs_hz,
        lining_material=lining_material,
        c0=C0,
        rho0=RHO0,
    )

    tl_rigid_cyl_db = rigid_cyl_duct.TL(Z_c=rigid_cyl_duct.Zc, omega=omega)
    tl_lined_cyl_db = circular_lined_duct.TL(Z_c=rigid_cyl_duct.Zc, omega=omega)

    plot_tl_circular_comparison(freqs_hz, tl_rigid_cyl_db, tl_lined_cyl_db)
    plt.show()
