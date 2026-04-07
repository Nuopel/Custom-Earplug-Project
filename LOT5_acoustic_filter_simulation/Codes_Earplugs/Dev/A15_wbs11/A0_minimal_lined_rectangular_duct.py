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

from toolkitsd.acoustmm import AcousticElement, RectangularDuct
from toolkitsd.porous import BasePorousMaterial, EquivalentFluidModel, MikiMaterial, MikiModel, build_porous_medium_props


def rigid_backed_surface_impedance_from_porous(
    material: BasePorousMaterial,
    frequencies_hz: np.ndarray,
    *,
    incidence_angle_deg: float = 90.0,
    model: EquivalentFluidModel | None = None,
) -> np.ndarray:
    """Return rigid-backed porous surface impedance for a locally reacting wall.

    For the lined-duct approximation used here, we take the porous wall as a
    locally reacting boundary and reuse the rigid-backed surface response from
    ``toolkitsd.porous``.
    """
    porous_props = build_porous_medium_props(material, frequencies_hz, model=model or MikiModel())
    response = porous_props.surface_response_on_rigid_backing(incidence_angle_deg=incidence_angle_deg)
    return np.asarray(response.surface_impedance, dtype=np.complex128).reshape(-1)


def calculate_kz_one_sided_rectangular_lined_duct(
    k0: np.ndarray,
    *,
    c0: float,
    rho0: float,
    zs_wall: np.ndarray,
    lined_span: float,
) -> np.ndarray:
    """Approximate axial wavenumber for a rectangular duct lined on one wall pair.

    Munjal, Section 6.4:
    - Eq. 6.59 gives the approximation of the transverse wavenumber
    - for one lined wall pair, the negative-sign branch is retained
    - kz then follows from the stationary-medium relation used in Eq. 6.64
    """
    q_wall = 1j * k0 * (rho0 * c0) * (lined_span / 2.0) / zs_wall
    # Munjal Eq. 6.59, with the negative-sign branch retained for one lined
    # wall pair ("only two opposite sides lined").
    kx2 = ((2.47 + q_wall - np.sqrt((2.47 + q_wall) ** 2 - 1.87 * q_wall)) / 0.38) * (4.0 / lined_span**2)
    return np.sqrt(k0**2 - kx2 + 0j)


def calculate_kz_double_sided_rectangular_lined_duct(
    k0: np.ndarray,
    *,
    c0: float,
    rho0: float,
    zs_x: np.ndarray,
    zs_y: np.ndarray,
    width_x: float,
    width_y: float,
) -> np.ndarray:
    """Approximate axial wavenumber for a rectangular duct lined on both wall pairs.

    Munjal, Section 6.4:
    - Eq. 6.59 is applied independently in x and y
    - Eq. 6.64 is used through alpha0 = -Im(kz)
    - the selected branch is the one that yields the lower attenuation
    """
    q = 1j * k0 * (rho0 * c0)
    qx = (q * (width_x / 2.0) / zs_x)[:, np.newaxis, np.newaxis]
    qy = (q * (width_y / 2.0) / zs_y)[:, np.newaxis, np.newaxis]
    signs = np.array([1.0, -1.0])

    kx2 = ((2.47 + qx + signs[np.newaxis, :, np.newaxis] * np.sqrt((2.47 + qx) ** 2 - 1.87 * qx)) / 0.38) * (
        4.0 / width_x**2
    )
    ky2 = ((2.47 + qy + signs[np.newaxis, np.newaxis, :] * np.sqrt((2.47 + qy) ** 2 - 1.87 * qy)) / 0.38) * (
        4.0 / width_y**2
    )

    kx_grid, ky_grid = np.broadcast_arrays(kx2, ky2)
    alpha0 = -np.imag(np.sqrt(k0[:, np.newaxis, np.newaxis] ** 2 - kx_grid - ky_grid + 0j))
    best_flat_idx = np.argmin(alpha0.reshape(k0.size, -1), axis=1)
    best_idx = [np.unravel_index(idx, alpha0.shape[1:]) for idx in best_flat_idx]

    chosen_kx2 = np.array([kx_grid[i, ix, iy] for i, (ix, iy) in enumerate(best_idx)], dtype=np.complex128)
    chosen_ky2 = np.array([ky_grid[i, ix, iy] for i, (ix, iy) in enumerate(best_idx)], dtype=np.complex128)
    return np.sqrt(k0**2 - chosen_kx2 - chosen_ky2 + 0j)


class LinedRectangularDuct(AcousticElement):
    """Minimal local example of a rectangular duct with effective lined-wall propagation.

    This is a script-local prototype to validate the model before integration in
    ``toolkitsd.acoustmm`` proper.
    """

    def __init__(
        self,
        *,
        width: float,
        height: float,
        length: float,
        frequencies_hz: np.ndarray,
        wall_mode: str = "one-sided",
        lining_material: BasePorousMaterial | None = None,
        lining_material_x: BasePorousMaterial | None = None,
        lining_material_y: BasePorousMaterial | None = None,
        zs_wall: np.ndarray | None = None,
        zs_x: np.ndarray | None = None,
        zs_y: np.ndarray | None = None,
        incidence_angle_deg: float = 90.0,
        porous_model: EquivalentFluidModel | None = None,
        c0: float = 343.0,
        rho0: float = 1.2,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.length = float(length)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.area = self.width * self.height
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64).ravel()
        self.omega = 2.0 * np.pi * self.frequencies_hz
        k0 = self.omega / self.c0

        if wall_mode == "one-sided":
            if zs_wall is None:
                if lining_material is None:
                    raise ValueError("Provide either zs_wall or lining_material for one-sided mode.")
                zs_wall = rigid_backed_surface_impedance_from_porous(
                    lining_material,
                    self.frequencies_hz,
                    incidence_angle_deg=incidence_angle_deg,
                    model=porous_model,
                )
            self.zs_wall = np.asarray(zs_wall, dtype=np.complex128).ravel()
            self.kz = calculate_kz_one_sided_rectangular_lined_duct(
                k0,
                c0=self.c0,
                rho0=self.rho0,
                zs_wall=self.zs_wall,
                lined_span=self.height,
            )
        elif wall_mode == "double-sided":
            if zs_x is None and lining_material_x is not None:
                zs_x = rigid_backed_surface_impedance_from_porous(
                    lining_material_x,
                    self.frequencies_hz,
                    incidence_angle_deg=incidence_angle_deg,
                    model=porous_model,
                )
            if zs_y is None and lining_material_y is not None:
                zs_y = rigid_backed_surface_impedance_from_porous(
                    lining_material_y,
                    self.frequencies_hz,
                    incidence_angle_deg=incidence_angle_deg,
                    model=porous_model,
                )
            if zs_x is None or zs_y is None:
                raise ValueError("Provide zs_x/zs_y or lining_material_x/lining_material_y for double-sided mode.")
            self.zs_x = np.asarray(zs_x, dtype=np.complex128).ravel()
            self.zs_y = np.asarray(zs_y, dtype=np.complex128).ravel()
            self.kz = calculate_kz_double_sided_rectangular_lined_duct(
                k0,
                c0=self.c0,
                rho0=self.rho0,
                zs_x=self.zs_x,
                zs_y=self.zs_y,
                width_x=self.width,
                width_y=self.height,
            )
        else:
            raise ValueError("wall_mode must be 'one-sided' or 'double-sided'.")
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


def plot_tl_comparison(freqs_hz: np.ndarray, tl_rigid_db: np.ndarray, tl_one_side_db: np.ndarray, tl_double_side_db: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs_hz, tl_rigid_db, linewidth=2.0, label="Rigid rectangular duct")
    plt.semilogx(freqs_hz, tl_one_side_db, linewidth=2.0, label="One-side lining")
    plt.semilogx(freqs_hz, tl_double_side_db, linewidth=2.0, label="Two-side lining")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Transmission Loss [dB]")
    plt.title("Minimal lined rectangular duct test")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="best")
if __name__ == "__main__":
    C0 = 343.0
    RHO0 = 1.2

    freqs_hz = np.logspace(np.log10(50.0), np.log10(5000.0), 300)
    omega = 2.0 * np.pi * freqs_hz

    width_x = 0.01
    width_y = 0.01
    length_duct = 0.1

    rigid_duct = RectangularDuct(width=width_x, height=width_y, length=length_duct, c0=C0, rho0=RHO0)

    # Minimal example wall material. The lining thickness here is the porous wall
    # depth used to compute the rigid-backed surface impedance Zs.
    lining_material_x = MikiMaterial(
        sigma=15000.0,
        thickness=0.005,
        rho0=RHO0,
        c0=C0,
        name="Miki lining X",
    )
    lining_material_y = MikiMaterial(
        sigma=15000.0,
        thickness=0.005,
        rho0=RHO0,
        c0=C0,
        name="Miki lining Y",
    )

    one_side_duct = LinedRectangularDuct(
        width=width_x,
        height=width_y,
        length=length_duct,
        frequencies_hz=freqs_hz,
        wall_mode="one-sided",
        lining_material=lining_material_y,
        c0=C0,
        rho0=RHO0,
    )
    double_side_duct = LinedRectangularDuct(
        width=width_x,
        height=width_y,
        length=length_duct,
        frequencies_hz=freqs_hz,
        wall_mode="double-sided",
        lining_material_x=lining_material_x,
        lining_material_y=lining_material_y,
        incidence_angle_deg=90.0,
        c0=C0,
        rho0=RHO0,
    )

    tl_rigid_db = rigid_duct.TL(Z_c=rigid_duct.Zc, omega=omega)
    tl_one_side_db = one_side_duct.TL(Z_c=rigid_duct.Zc, omega=omega)
    tl_double_side_db = double_side_duct.TL(Z_c=rigid_duct.Zc, omega=omega)

    plot_tl_comparison(freqs_hz, tl_rigid_db, tl_one_side_db, tl_double_side_db)
    plt.show()
