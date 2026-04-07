"""
Measurement and estimation utilities (doublet/monopole plane-wave techniques).
"""

from __future__ import annotations

import numpy as np

class BaseMeasurement:
    """Shared helpers for measurement estimators."""

    @staticmethod
    def _apply_time_convention(values: np.ndarray, time_convention: str) -> np.ndarray:
        """Conjugate data when using the +jwt convention."""
        return np.conj(values) if time_convention == "jwt" else values

    @staticmethod
    def _normalize_pressures(pressures: np.ndarray, expected_mics: int | None = None) -> np.ndarray:
        """Ensure pressure array is (n_freq, n_mics)."""
        p = np.asarray(pressures)
        if p.ndim == 3 and p.shape[2] == 1:
            p = p[:, :, 0]
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        if p.ndim != 2:
            raise ValueError("pressures must be 2-D [n_freq, n_mics]")
        if expected_mics is not None and p.shape[1] != expected_mics:
            raise ValueError(f"pressures must have {expected_mics} microphones; got {p.shape[1]}")
        return p

    @staticmethod
    def _normalize_mics(mic_positions: np.ndarray, expected_mics: int | None = None) -> np.ndarray:
        """Ensure mic coordinates are (3, n_mics)."""
        mics = np.asarray(mic_positions)
        if mics.ndim != 2 or mics.shape[0] != 3:
            raise ValueError("mic_positions must be shaped (3, n_mics)")
        if expected_mics is not None and mics.shape[1] != expected_mics:
            raise ValueError(f"mic_positions must have {expected_mics} microphones; got {mics.shape[1]}")
        return mics

    @staticmethod
    def _incidence_from_normal(elevation_deg: float) -> float:
        """Return sin(theta) for an elevation measured from the ground plane."""
        return np.sin(np.deg2rad(elevation_deg))

    @staticmethod
    def _dz(mic_positions: np.ndarray) -> float:
        """Spacing between the first two microphones along z."""
        if mic_positions.shape[1] < 2:
            raise ValueError("At least two microphones are required to compute spacing.")
        return abs(mic_positions[2, 0] - mic_positions[2, 1])


class TwoMicPlaneWave(BaseMeasurement):
    """Two-microphone plane-wave estimators."""

    @staticmethod
    def pv(
        pressures: np.ndarray,
        mic_positions: np.ndarray,
        omega: np.ndarray,
        incidence_deg: float,
        z0: float,
        c0: float,
        mic_height: float,
        time_convention: str = "jwt",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pressure/velocity formulation."""
        pressures = BaseMeasurement._apply_time_convention(pressures, time_convention)
        p = BaseMeasurement._normalize_pressures(pressures, expected_mics=2)
        mics = BaseMeasurement._normalize_mics(mic_positions, expected_mics=2)
        dz = BaseMeasurement._dz(mics)

        sin_inc = BaseMeasurement._incidence_from_normal(incidence_deg)
        v_doublet = (p[:, 0] - p[:, 1]) / (z0 / c0 * 1j * omega * dz)
        z_doublet = (p[:, 0] + p[:, 1]) / 2 / v_doublet

        cts = z0 / sin_inc
        r_pw = (z_doublet - cts) / (z_doublet + cts) * np.exp(
            -2j * omega / c0 * sin_inc * (dz / 2 + mic_height)
        )
        r_pw = BaseMeasurement._apply_time_convention(r_pw, time_convention)
        zs_pw = (1 + r_pw) / (1 - r_pw) * cts
        alpha_pw = 1 - np.abs(r_pw) ** 2
        return alpha_pw, zs_pw, r_pw

    @staticmethod
    def transfer(
        pressures: np.ndarray,
        mic_positions: np.ndarray,
        omega: np.ndarray,
        incidence_deg: float,
        z0: float,
        c0: float,
        mic_height: float,
        time_convention: str = "jwt",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transfer-function formulation (Allard 1985)."""
        pressures = BaseMeasurement._apply_time_convention(pressures, time_convention)
        p = BaseMeasurement._normalize_pressures(pressures, expected_mics=2)
        mics = BaseMeasurement._normalize_mics(mic_positions, expected_mics=2)
        dz = BaseMeasurement._dz(mics)

        sin_inc = BaseMeasurement._incidence_from_normal(incidence_deg)
        h_12 = p[:, 1] / p[:, 0]

        r_pw = (h_12 - np.exp(-1j * omega / c0 * dz * sin_inc)) / (
            np.exp(1j * omega / c0 * dz * sin_inc) - h_12
        ) * np.exp(-2j * omega / c0 * mic_height * sin_inc)
        r_pw = BaseMeasurement._apply_time_convention(r_pw, time_convention)

        zs_pw = z0 / sin_inc * (1 + r_pw) / (1 - r_pw)
        alpha_pw = 1 - np.abs(r_pw) ** 2
        return alpha_pw, zs_pw, r_pw


class OneMicPlaneWave(BaseMeasurement):
    """Single-microphone plane-wave estimator."""

    @staticmethod
    def estimate(
        p_material: np.ndarray,
        p_rigid: np.ndarray,
        mic_position: np.ndarray,
        k: np.ndarray,
        elevation_deg: float,
        z0: float,
        z_ground: float,
        time_convention: str = "jwt",
    ) -> tuple[np.ndarray, np.ndarray]:
        p_material = BaseMeasurement._apply_time_convention(p_material, time_convention)
        p_rigid = BaseMeasurement._apply_time_convention(p_rigid, time_convention)
        p_material = BaseMeasurement._normalize_pressures(p_material, expected_mics=1)
        p_rigid = BaseMeasurement._normalize_pressures(p_rigid, expected_mics=1)
        mic = BaseMeasurement._normalize_mics(mic_position, expected_mics=1)

        sin_inc = BaseMeasurement._incidence_from_normal(elevation_deg)
        k_z = k * sin_inc
        prop = np.exp(-2j * k_z.reshape(-1, 1) * mic[2, :])
        rinf = np.exp(-1j * k_z.reshape(-1, 1) * (-z_ground))

        r_coeff = (p_material / p_rigid * (prop + rinf) - prop).squeeze()
        r_coeff = BaseMeasurement._apply_time_convention(r_coeff, time_convention)
        zs_pw = (1 + r_coeff) / (1 - r_coeff) * z0 / sin_inc
        alpha_pw = 1 - np.abs(r_coeff) ** 2
        return alpha_pw, zs_pw


class TwoMicSpherical(BaseMeasurement):
    """Two-microphone spherical-wave estimator (image source approximation)."""

    @staticmethod
    def transfer(
        pressures: np.ndarray,
        mic_positions: np.ndarray,
        k: np.ndarray,
        source: np.ndarray,
        elevation_deg: float,
        z0: float,
        time_convention: str = "jwt",
        porous_height: float = 0,
        image_source: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transfer-function formulation with spherical wavefronts.

        Args:
            pressures: Complex pressures ``[F, 2]`` or ``[F, 2, 1]`` at the two mics.
            mic_positions: Microphone coordinates ``[3, 2]``.
            k: Wavenumbers ``[F,]``.
            source: Source coordinates ``[3,]`` or ``[3, 1]``.
            image_source: Mirrored source coordinates ``[3,]`` or ``[3, 1]``.
            elevation_deg: Incidence elevation from the ground plane (deg).
            porous_height: Porous layer height offset used in the impedance model.
            z0: Characteristic impedance of air.
            time_convention: ``"jwt"`` or ``"neg_jwt"``.
        """

        # pressures = BaseMeasurement._apply_time_convention(pressures, time_convention)
        p = BaseMeasurement._normalize_pressures(pressures, expected_mics=2)
        mics = BaseMeasurement._normalize_mics(mic_positions, expected_mics=2)

        if image_source is None:
            image_source = source.copy()
            image_source[2, :] *= -1

        if time_convention == "jwt" :
                p = np.copy(np.conj(p))

        source = np.asarray(source).reshape(3, -1)
        image_source = np.asarray(image_source).reshape(3, -1)
        k = np.asarray(k).reshape(-1)

        h_m = p[:, 1] / p[:, 0]

        time_sign = 1
        from toolkitsd.acoustic.pressure_gen import MonopoleSource
        p_incident = MonopoleSource.compute_monopole_pressure(
            receivers=mics,
            source_positions=source,
            wavenumbers=k,
            time_sign=time_sign,
        )[:, :, 0]

        p_image = MonopoleSource.compute_monopole_pressure(
            receivers=mics,
            source_positions=image_source,
            wavenumbers=k,
            time_sign=time_sign,
        )[:, :, 0]

        r_xy_sq = (source[0] - mics[0, 0]) ** 2 + (source[1] - mics[1, 0]) ** 2

        e11_r2 = p_incident[:, 1]
        e12_r2p = p_image[:, 1]

        e21_r1 = p_incident[:, 0]
        e22_r1p = p_image[:, 0]

        r_pm2 = (e11_r2 - h_m * e21_r1) / (h_m * e22_r1p - e12_r2p)

        costheta = (source[2] - porous_height) / np.sqrt(r_xy_sq + (source[2] - porous_height) ** 2)
        cts2 = 1 / (1 + 1j * costheta / (k * source[2])) * z0 / costheta
        zs_sph = (1 + r_pm2) / (1 - r_pm2) * cts2

        r_sph = (zs_sph - z0 / np.sin(np.deg2rad(elevation_deg))) / (zs_sph + z0 / np.sin(np.deg2rad(elevation_deg)))
        # r_sph = BaseMeasurement._apply_time_convention(r_sph, time_convention)
        alpha_sph = 1 - np.abs(r_sph) ** 2

        if time_convention == "jwt" :
            return np.conj(alpha_sph), np.conj(zs_sph),  np.conj(r_sph)
        return alpha_sph, zs_sph, r_sph


def two_mic_plane_wave_pv(
    pressures: np.ndarray,
    mic_positions: np.ndarray,
    omega: np.ndarray,
    incidence_deg: float,
    z0: float,
    c0: float,
    mic_height: float,
    time_convention: str = "jwt",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin wrapper over TwoMicPlaneWave.pv for backward compatibility."""
    return TwoMicPlaneWave.pv(
        pressures=pressures,
        mic_positions=mic_positions,
        omega=omega,
        incidence_deg=incidence_deg,
        z0=z0,
        c0=c0,
        mic_height=mic_height,
        time_convention=time_convention,
    )


def two_mic_plane_wave_transfer(
    pressures: np.ndarray,
    mic_positions: np.ndarray,
    omega: np.ndarray,
    incidence_deg: float,
    z0: float,
    c0: float,
    mic_height: float,
    time_convention: str = "jwt",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin wrapper over TwoMicPlaneWave.transfer for backward compatibility."""
    return TwoMicPlaneWave.transfer(
        pressures=pressures,
        mic_positions=mic_positions,
        omega=omega,
        incidence_deg=incidence_deg,
        z0=z0,
        c0=c0,
        mic_height=mic_height,
        time_convention=time_convention,
    )


def one_mic_plane_wave(
    p_material: np.ndarray,
    p_rigid: np.ndarray,
    mic_position: np.ndarray,
    k: np.ndarray,
    elevation_deg: float,
    z0: float,
    z_ground: float,
    time_convention: str = "jwt",
) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper over OneMicPlaneWave.estimate for backward compatibility."""
    return OneMicPlaneWave.estimate(
        p_material=p_material,
        p_rigid=p_rigid,
        mic_position=mic_position,
        k=k,
        elevation_deg=elevation_deg,
        z0=z0,
        z_ground=z_ground,
        time_convention=time_convention,
    )


def two_mic_spherical_wave_transfer(
    pressures: np.ndarray,
    mic_positions: np.ndarray,
    k: np.ndarray,
    source: np.ndarray,
    image_source: np.ndarray,
    elevation_deg: float,
    porous_height: float,
    z0: float,
    time_convention: str = "jwt",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin wrapper over TwoMicSpherical.transfer for backward compatibility."""
    return TwoMicSpherical.transfer(
        pressures=pressures,
        mic_positions=mic_positions,
        k=k,
        source=source,
        image_source=image_source,
        elevation_deg=elevation_deg,
        porous_height=porous_height,
        z0=z0,
        time_convention=time_convention,
    )
