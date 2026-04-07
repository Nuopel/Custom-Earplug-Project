"""Frozen transfer-matrix helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .base import WaveguideElement


class FrozenMatrixElement(WaveguideElement):
    """Waveguide element backed by a precomputed matrix stack."""

    def __init__(self, matrices: np.ndarray, *, state_basis: Literal["pu", "pv"] = "pu") -> None:
        matrices = np.asarray(matrices, dtype=np.complex128)
        if matrices.ndim != 3 or matrices.shape[1:] != (2, 2):
            raise ValueError(f"matrices must have shape (N, 2, 2), got {matrices.shape}")
        if state_basis not in {"pu", "pv"}:
            raise ValueError(f"Unsupported state_basis {state_basis!r}")
        self.matrices = matrices
        self.state_basis = state_basis

    @staticmethod
    def _validate_area(area_m2: float) -> float:
        area_m2 = float(area_m2)
        if area_m2 <= 0.0:
            raise ValueError(f"area_m2 must be strictly positive, got {area_m2}")
        return area_m2

    @staticmethod
    def _matrix_pv_to_pu(matrix_pv: np.ndarray, area_m2: float) -> np.ndarray:
        matrix_pu = np.array(matrix_pv, dtype=np.complex128, copy=True)
        matrix_pu[:, 0, 1] = matrix_pu[:, 0, 1] / area_m2
        matrix_pu[:, 1, 0] = matrix_pu[:, 1, 0] * area_m2
        return matrix_pu

    @staticmethod
    def _matrix_pu_to_pv(matrix_pu: np.ndarray, area_m2: float) -> np.ndarray:
        matrix_pv = np.array(matrix_pu, dtype=np.complex128, copy=True)
        matrix_pv[:, 0, 1] = matrix_pv[:, 0, 1] * area_m2
        matrix_pv[:, 1, 0] = matrix_pv[:, 1, 0] / area_m2
        return matrix_pv

    @classmethod
    def from_pu(cls, matrices: np.ndarray) -> "FrozenMatrixElement":
        return cls(matrices, state_basis="pu")

    @classmethod
    def from_pv(cls, matrices: np.ndarray) -> "FrozenMatrixElement":
        return cls(matrices, state_basis="pv")

    @classmethod
    def from_pv_converted_to_pu(cls, matrices: np.ndarray, area_m2: float) -> "FrozenMatrixElement":
        area_m2 = cls._validate_area(area_m2)
        return cls(cls._matrix_pv_to_pu(matrices, area_m2), state_basis="pu")

    @classmethod
    def from_pu_converted_to_pv(cls, matrices: np.ndarray, area_m2: float) -> "FrozenMatrixElement":
        area_m2 = cls._validate_area(area_m2)
        return cls(cls._matrix_pu_to_pv(matrices, area_m2), state_basis="pv")

    def to_pu(self, area_m2: float) -> "FrozenMatrixElement":
        if self.state_basis == "pu":
            return FrozenMatrixElement(self.matrices.copy(), state_basis="pu")
        area_m2 = self._validate_area(area_m2)
        return FrozenMatrixElement(self._matrix_pv_to_pu(self.matrices, area_m2), state_basis="pu")

    def to_pv(self, area_m2: float) -> "FrozenMatrixElement":
        if self.state_basis == "pv":
            return FrozenMatrixElement(self.matrices.copy(), state_basis="pv")
        area_m2 = self._validate_area(area_m2)
        return FrozenMatrixElement(self._matrix_pu_to_pv(self.matrices, area_m2), state_basis="pv")

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        if self.matrices.shape[0] != omega.size:
            raise ValueError(
                f"matrix stack and omega must have same length, got {self.matrices.shape[0]} and {omega.size}"
            )
        return self.matrices


def _as_frequency_parameter(value: np.ndarray | complex | float, omega: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 0:
        return np.full(omega.shape, arr, dtype=np.complex128)
    arr = arr.ravel()
    if arr.size != omega.size:
        raise ValueError(f"{name} and omega must have same length, got {arr.size} and {omega.size}")
    return arr


class EquivalentDuct(WaveguideElement):
    """Homogeneous equivalent duct defined directly by effective k and Zc."""

    def __init__(self, length: float, k_eff: np.ndarray | complex | float, zc_eff: np.ndarray | complex | float) -> None:
        if length <= 0.0:
            raise ValueError("length must be positive")
        self.length = float(length)
        self.k_eff = k_eff
        self.zc_eff = zc_eff

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        k_eff = _as_frequency_parameter(self.k_eff, omega, "k_eff")
        zc_eff = _as_frequency_parameter(self.zc_eff, omega, "zc_eff")

        kL = k_eff * self.length
        ckL = np.cos(kL)
        skL = np.sin(kL)

        T = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = ckL
        T[:, 0, 1] = 1j * zc_eff * skL
        T[:, 1, 0] = 1j * skL / zc_eff
        T[:, 1, 1] = ckL
        return T


class EquivalentSeriesImpedance(WaveguideElement):
    """Frozen series element defined directly by an effective series impedance."""

    def __init__(self, z_series: np.ndarray | complex | float) -> None:
        self.z_series = z_series

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        z_series = _as_frequency_parameter(self.z_series, omega, "z_series")
        T = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = 1.0
        T[:, 0, 1] = z_series
        T[:, 1, 1] = 1.0
        return T


class EquivalentParallelImpedance(WaveguideElement):
    """Frozen parallel element defined directly by an effective parallel impedance."""

    def __init__(self, z_parallel: np.ndarray | complex | float) -> None:
        self.z_parallel = z_parallel

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        z_parallel = _as_frequency_parameter(self.z_parallel, omega, "z_parallel")
        y_parallel = 1.0 / z_parallel
        T = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = 1.0
        T[:, 1, 0] = y_parallel
        T[:, 1, 1] = 1.0
        return T
