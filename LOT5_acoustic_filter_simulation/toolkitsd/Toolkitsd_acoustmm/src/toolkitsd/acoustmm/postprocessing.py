"""Post-processing helpers for identified transfer-matrix workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .acoustic_params import AcousticParameters

from .elements.boundaries import RadiationImpedance
from .elements.frozen import FrozenMatrixElement


@dataclass(frozen=True)
class GeometryConfig:
    l1: float
    l2: float
    l_slab: float
    l_cav: float
    l_load_a: float
    l_load_b: float
    r_tube: float
    r_slab: float

    @property
    def s_tube(self) -> float:
        return np.pi * self.r_tube**2

    @property
    def s_slab(self) -> float:
        return np.pi * self.r_slab**2


@dataclass(frozen=True)
class ThreeMicPostProcessor:
    params: AcousticParameters
    geometry: GeometryConfig

    def _broadcast_impedance(self, impedance: np.ndarray | complex | float, k: np.ndarray) -> np.ndarray:
        impedance = np.asarray(impedance, dtype=np.complex128)
        if impedance.ndim == 0:
            impedance = np.full(k.shape, impedance, dtype=np.complex128)
        return np.broadcast_to(impedance, k.shape)

    def _earcanal_coefficients(
        self,
        identified_matrix_ep_pu: np.ndarray,
        *,
        z_r: np.ndarray | complex | float,
        z_tm: np.ndarray | complex | float,
        z_ec: np.ndarray | complex | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k = self.params.wavenumbers
        z_r = self._broadcast_impedance(z_r, k)
        z_tm = self._broadcast_impedance(z_tm, k)
        z_ec = self._broadcast_impedance(z_ec, k)

        a = identified_matrix_ep_pu[:, 0, 0]
        b = identified_matrix_ep_pu[:, 0, 1]
        c = identified_matrix_ep_pu[:, 1, 0]
        d = identified_matrix_ep_pu[:, 1, 1]

        denom = a + b / z_ec + c * z_ec + d
        r_ep = (a + b / z_ec - c * z_ec - d) / denom
        tau_ep = 2.0 * np.exp(1j * k * self.geometry.l_slab) / denom
        r_r = (z_r - z_ec) / (z_r + z_ec)
        r_tm = (z_tm - z_ec) / (z_tm + z_ec)
        return z_ec, r_ep, tau_ep, r_r, r_tm

    def compute_reduced_il_from_matrix(
        self,
        identified_matrix_ep_pu: np.ndarray,
        *,
        z_r: np.ndarray | complex | float,
        z_tm: np.ndarray | complex | float,
        z_ec: np.ndarray | complex | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = self.params.wavenumbers


        z_ec, r_ep, tau_ep, r_r, r_tm = self._earcanal_coefficients(
            identified_matrix_ep_pu,
            z_r=z_r,
            z_tm=z_tm,
            z_ec=z_ec,
        )

        l_ec = self.geometry.l_slab + self.geometry.l_cav
        l_id = self.geometry.l_slab

        tl_ep_db = -20.0 * np.log10(np.maximum(np.abs(tau_ep), np.finfo(float).tiny))
        cavity_factor = (
            2.0
            * abs(z_ec)
            / abs(z_r + z_ec)
            * (1.0 - r_ep * r_tm * np.exp(-2j * k * (l_ec - l_id)))
            / (1.0 - r_r * r_tm * np.exp(-2j * k * l_ec))
        )
        ilc_db = 20.0 * np.log10(np.maximum(np.abs(cavity_factor), np.finfo(float).tiny))
        il_db = tl_ep_db + ilc_db
        return tl_ep_db, ilc_db, il_db

    def compute_tm_pressure_il_from_matrix(
        self,
        identified_matrix_ep_pu: np.ndarray,
        *,
        z_r: np.ndarray | complex | float,
        z_tm: np.ndarray | complex | float,
        z_ec: np.ndarray | complex | float,
        p0: np.ndarray | complex | float = 1.0,
        x: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = self.params.wavenumbers
        z_ec, r_ep, tau_ep, r_r, r_tm = self._earcanal_coefficients(
            identified_matrix_ep_pu,
            z_r=z_r,
            z_tm=z_tm,
            z_ec=z_ec,
        )
        p0 = self._broadcast_impedance(p0, k)

        l_ec = self.geometry.l_slab + self.geometry.l_cav
        l_id = self.geometry.l_slab
        if x is None:
            x = l_ec

        common_numerator = np.exp(-1j * k * x) + r_tm * np.exp(-2j * k * l_ec) * np.exp(1j * k * x)
        a_open = 2.0 * p0 * z_ec / (z_r + z_ec)
        a_occl = tau_ep * p0

        p_open = a_open * common_numerator / (1.0 - r_r * r_tm * np.exp(-2j * k * l_ec))
        p_occl = a_occl * common_numerator / (1.0 - r_ep * r_tm * np.exp(-2j * k * (l_ec - l_id)))
        il_db = 20.0 * np.log10(np.maximum(np.abs(p_open), np.finfo(float).tiny) / np.maximum(np.abs(p_occl), np.finfo(float).tiny))
        return p_open, p_occl, il_db

    def identify_transfer_matrix_two_loads(
        self,
        p0_a: np.ndarray,
        v0_a: np.ndarray,
        pl_a: np.ndarray,
        vl_a: np.ndarray,
        p0_b: np.ndarray,
        v0_b: np.ndarray,
        pl_b: np.ndarray,
        vl_b: np.ndarray,
    ) -> np.ndarray:
        delta = pl_a * vl_b - pl_b * vl_a

        identified_matrix = np.empty((p0_a.size, 2, 2), dtype=np.complex128)
        identified_matrix[:, 0, 0] = (p0_a * vl_b - p0_b * vl_a) / delta
        identified_matrix[:, 0, 1] = (p0_b * pl_a - p0_a * pl_b) / delta
        identified_matrix[:, 1, 0] = (v0_a * vl_b - v0_b * vl_a) / delta
        identified_matrix[:, 1, 1] = (pl_a * v0_b - pl_b * v0_a) / delta
        return identified_matrix

    def identify_transfer_element_from_h_two_loads(
        self,
        h12: np.ndarray,
        h13: np.ndarray,
        *,
        k_tube: np.ndarray,
        z_tube: np.ndarray,
        l1: float,
        l2_by_load: np.ndarray,
        l3_by_load: np.ndarray,
        s_tube: float,
        s_eff: float,
        k_ec: np.ndarray | None = None,
        return_basis: Literal["pu", "pv"] = "pu",
    ) -> FrozenMatrixElement:
        p0_a, v0_a, pl_a, vl_a, p0_b, v0_b, pl_b, vl_b = self.reconstruct_boundary_states_from_h(
            h12,
            h13,
            k_tube=k_tube,
            z_tube=z_tube,
            l1=l1,
            l2_by_load=l2_by_load,
            l3_by_load=l3_by_load,
            s_tube=s_tube,
            s_eff=s_eff,
            k_ec=k_ec,
        )
        identified_matrix_pv = self.identify_transfer_matrix_two_loads(
            p0_a,
            v0_a,
            pl_a,
            vl_a,
            p0_b,
            v0_b,
            pl_b,
            vl_b,
        )
        if return_basis == "pv":
            return FrozenMatrixElement.from_pv(identified_matrix_pv)
        if return_basis == "pu":
            return FrozenMatrixElement.from_pv_converted_to_pu(identified_matrix_pv, s_eff)
        raise ValueError(f"Unsupported return_basis {return_basis!r}")

    def reconstruct_boundary_states_from_h(
        self,
        h12: np.ndarray,
        h13: np.ndarray,
        *,
        k_tube: np.ndarray,
        z_tube: np.ndarray,
        l1: float,
        l2_by_load: np.ndarray,
        l3_by_load: np.ndarray,
        s_tube: float,
        s_eff: float,
        k_ec: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h12 = np.asarray(h12, dtype=np.complex128)
        h13 = np.asarray(h13, dtype=np.complex128)
        k_tube = np.asarray(k_tube, dtype=np.complex128)
        z_tube = np.asarray(z_tube, dtype=np.complex128)
        l2_by_load = np.asarray(l2_by_load, dtype=float)
        l3_by_load = np.asarray(l3_by_load, dtype=float)

        if h12.shape != h13.shape:
            raise ValueError(f"h12 and h13 must have the same shape, got {h12.shape} and {h13.shape}")
        if h12.ndim != 2 or h12.shape[1] != 2:
            raise ValueError(f"h12 and h13 must have shape (n_freq, 2), got {h12.shape}")
        if k_tube.shape != (h12.shape[0],):
            raise ValueError(f"k_tube must have shape ({h12.shape[0]},), got {k_tube.shape}")
        if z_tube.shape != (h12.shape[0],):
            raise ValueError(f"z_tube must have shape ({h12.shape[0]},), got {z_tube.shape}")
        if l2_by_load.shape != (2,):
            raise ValueError(f"l2_by_load must have shape (2,), got {l2_by_load.shape}")
        if l3_by_load.shape != (2,):
            raise ValueError(f"l3_by_load must have shape (2,), got {l3_by_load.shape}")

        if k_ec is None:
            k_ec = k_tube
        else:
            k_ec = np.asarray(k_ec, dtype=np.complex128)
            if k_ec.shape != (h12.shape[0],):
                raise ValueError(f"k_ec must have shape ({h12.shape[0]},), got {k_ec.shape}")

        k_t = k_tube[:, None]
        z_t = z_tube[:, None]
        k_e = k_ec[:, None]
        l2 = l2_by_load[None, :]
        l3 = l3_by_load[None, :]

        denom = h12 * np.exp(-1j * k_t * l1) - 1.0

        p0 = -1j*2.0 * np.exp(1j * k_t * l2) * (h12 * np.sin(k_t * (l1 + l2)) - np.sin(k_t * l2)) / denom
        v0 = (
            (s_tube / s_eff)
            * (1.0 / z_t)
            * 2.0
            * np.exp(1j * k_t * l2)
            * (h12 * np.cos(k_t * (l1 + l2)) - np.cos(k_t * l2))
            / denom
        )
        pl = -1j*2.0 * np.exp(1j * k_t * l2) * (h13 * np.sin(k_t * l1) * np.cos(k_e * l3)) / denom
        vl = (1.0 / z_t) * 2.0 * np.exp(1j * k_t * l2) * (h13 * np.sin(k_t * l1) * np.sin(k_t * l3)) / denom

        p0_a, p0_b = p0[:, 0], p0[:, 1]
        v0_a, v0_b = v0[:, 0], v0[:, 1]
        pl_a, pl_b = pl[:, 0], pl[:, 1]
        vl_a, vl_b = vl[:, 0], vl[:, 1]
        return p0_a, v0_a, pl_a, vl_a, p0_b, v0_b, pl_b, vl_b

    def flanged_piston_radiation(self, omega: np.ndarray) -> np.ndarray:
        return RadiationImpedance(radius=self.geometry.r_slab, mode="flanged", c0=self.params.c0, rho0=self.params.rho0).Z(omega)
