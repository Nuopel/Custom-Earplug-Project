"""Base abstractions for acoustmm models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np


class AcousticModel(ABC):
    """Minimal shared root for acoustical models."""

    @staticmethod
    def _as_omega_array(omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")
        return omega


@dataclass(frozen=True)
class EquivalentDuctRetrievalResult:
    """Equivalent homogeneous duct retrieved from a 2-port waveguide system."""

    duct: "EquivalentDuct"
    Z_eff: np.ndarray
    k_eff: np.ndarray
    T_prim: np.ndarray
    rho_eff: np.ndarray | None = None
    K_eff: np.ndarray | None = None
    rho_eff_lf: np.ndarray | None = None
    K_eff_lf: np.ndarray | None = None


class WaveguideElement(AcousticModel, ABC):
    """2x2 transfer-matrix element using the state vector ``[p, U]``.

    ``WaveguideElement`` is the primary abstraction for 1D cascaded models:
    ducts, slab approximations, junctions, porous sections, and similar
    waveguide-confined elements. ``AcousticElement`` is kept as a
    compatibility alias below.
    """

    state_basis: Literal["pu", "pv"] = "pu"

    @abstractmethod
    def matrix(self, omega: np.ndarray) -> np.ndarray:
        """Return the transfer matrix with shape ``(N_freq, 2, 2)``."""

    def _require_compatible_state_basis(self, other: "WaveguideElement", *, operation: str) -> None:
        if self.state_basis != other.state_basis:
            raise ValueError(
                f"Cannot {operation} waveguide elements with incompatible state bases: "
                f"{self.state_basis!r} and {other.state_basis!r}"
            )

    def __add__(self, other: "WaveguideElement") -> "ComposedElement":
        if not isinstance(other, WaveguideElement):
            return NotImplemented
        self._require_compatible_state_basis(other, operation="cascade")
        return ComposedElement(self, other)

    def in_parallel_with(self, *others: "WaveguideElement") -> "ParallelElement":
        if len(others) == 0:
            raise ValueError("in_parallel_with requires at least one other WaveguideElement")
        for other in others:
            if not isinstance(other, WaveguideElement):
                raise TypeError(f"Cannot compose object of type {type(other)!r} in parallel with WaveguideElement")
            self._require_compatible_state_basis(other, operation="compose in parallel")
            if self.state_basis != "pu":
                raise ValueError(
                    "Parallel composition is currently only implemented for WaveguideElement "
                    "objects in the 'pu' state basis"
                )
        return ParallelElement(self, *others)

    def __floordiv__(self, other: "WaveguideElement") -> "ParallelElement":
        if not isinstance(other, WaveguideElement):
            return NotImplemented
        return self.in_parallel_with(other)

    def decascade_right(
        self,
        other: "WaveguideElement",
        *,
        method: Literal["direct", "tikhonov", "lcurve"] = "direct",
        regularization: float | None = None,
        lambda_grid: np.ndarray | None = None,
    ) -> "DecascadedElement":
        if not isinstance(other, WaveguideElement):
            raise TypeError(f"Cannot decascade object of type {type(other)!r} from WaveguideElement")
        self._require_compatible_state_basis(other, operation="decascade")
        return DecascadedElement(
            self,
            other,
            side="right",
            method=method,
            regularization=regularization,
            lambda_grid=lambda_grid,
        )

    def decascade_left(
        self,
        other: "WaveguideElement",
        *,
        method: Literal["direct", "tikhonov", "lcurve"] = "direct",
        regularization: float | None = None,
        lambda_grid: np.ndarray | None = None,
    ) -> "DecascadedElement":
        if not isinstance(other, WaveguideElement):
            raise TypeError(f"Cannot decascade object of type {type(other)!r} from WaveguideElement")
        self._require_compatible_state_basis(other, operation="decascade")
        return DecascadedElement(
            self,
            other,
            side="left",
            method=method,
            regularization=regularization,
            lambda_grid=lambda_grid,
        )

    def __sub__(self, other: "WaveguideElement") -> "DecascadedElement":
        return self.decascade_right(other)

    def __radd__(self, other: object) -> "WaveguideElement":
        if other == 0:
            return self
        if isinstance(other, WaveguideElement):
            return other.__add__(self)
        raise TypeError(f"Cannot add object of type {type(other)!r} to WaveguideElement")

    def Z_in(self, Z_load: np.ndarray | complex | float, omega: np.ndarray) -> np.ndarray:
        """Input impedance at the element inlet for downstream load ``Z_load``."""
        omega = self._as_omega_array(omega)
        T = self.matrix(omega)
        load = np.asarray(Z_load, dtype=np.complex128)
        if load.ndim == 0:
            load = np.full(T.shape[0], load, dtype=np.complex128)
        load = np.broadcast_to(load, (T.shape[0],))

        out = np.empty(T.shape[0], dtype=np.complex128)
        inf_mask = np.isinf(load)
        if np.any(inf_mask):
            out[inf_mask] = T[inf_mask, 0, 0] / T[inf_mask, 1, 0]
        if np.any(~inf_mask):
            num = T[~inf_mask, 0, 0] * load[~inf_mask] + T[~inf_mask, 0, 1]
            den = T[~inf_mask, 1, 0] * load[~inf_mask] + T[~inf_mask, 1, 1]
            out[~inf_mask] = num / den
        return out

    def TL(self, Z_c: float | complex | np.ndarray, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        T = self.matrix(omega)
        Z_c = np.asarray(Z_c)
        num = T[:, 0, 0] + T[:, 0, 1] / Z_c + T[:, 1, 0] * Z_c + T[:, 1, 1]
        return 20.0 * np.log10(np.abs(num / 2.0))

    def scattering_coefficients(
            self,
            Z_in: np.ndarray | complex | float,
            Z_out: np.ndarray | complex | float | None,
            omega: np.ndarray,
            *,
            k_ref: np.ndarray | complex | float | None = None,
            length: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the 2-port scattering coefficients (S11, S21, S12, S22)
        from the PU transfer matrix.

        Parameters
        ----------
        Z_in :
            Reference characteristic impedance at port 1 (input side).
        Z_out :
            Reference characteristic impedance at port 2 (output side).
            If None, the equal-reference case is assumed: Z_out = Z_in.
        omega :
            Angular frequency array.
        k_ref, length :
            Optional phase-reference normalization applied to transmission
            coefficients S21 and S12 through exp(-1j * k_ref * length).

        Notes
        -----
        For the transfer matrix convention

            [p1]   [A B] [p2]
            [U1] = [C D] [U2]

        the unequal-reference scattering formulas are:

            den = A + B/Z_out + C*Z_in + D*(Z_in/Z_out)

            S11 = (A + B/Z_out - C*Z_in - D*(Z_in/Z_out)) / den
            S21 = 2 / den
            S22 = (-A + B/Z_out - C*Z_in + D*(Z_in/Z_out)) / den
            S12 = 2*det(T)*(Z_in/Z_out) / den

        When Z_in == Z_out, this reduces to the standard equal-reference formulas.
        """
        omega = self._as_omega_array(omega)
        T = self.matrix(omega)

        n_freq = T.shape[0]

        Z_in = np.asarray(Z_in, dtype=np.complex128)
        if Z_in.ndim == 0:
            Z_in = np.full(n_freq, Z_in, dtype=np.complex128)
        Z_in = np.broadcast_to(Z_in, (n_freq,))

        if Z_out is None:
            Z_out = Z_in
        else:
            Z_out = np.asarray(Z_out, dtype=np.complex128)
            if Z_out.ndim == 0:
                Z_out = np.full(n_freq, Z_out, dtype=np.complex128)
            Z_out = np.broadcast_to(Z_out, (n_freq,))

        A = T[:, 0, 0]
        B = T[:, 0, 1]
        C = T[:, 1, 0]
        D = T[:, 1, 1]

        det_t = A * D - B * C
        ratio = Z_in / Z_out
        den = A + B / Z_out + C * Z_in + D * ratio

        if (k_ref is None) ^ (length is None):
            raise ValueError(
                "k_ref and length must be provided together for phase-normalized transmission"
            )

        phase = 1.0
        if k_ref is not None and length is not None:
            k_ref = np.asarray(k_ref, dtype=np.complex128)
            if k_ref.ndim == 0:
                k_ref = np.full(n_freq, k_ref, dtype=np.complex128)
            k_ref = np.broadcast_to(k_ref, (n_freq,))
            phase = np.exp(-1j * k_ref * length)

        s11 = (A + B / Z_out - C * Z_in - D * ratio) / den
        s21 = 2.0 * phase / den
        s22 = (-A + B / Z_out - C * Z_in + D * ratio) / den
        s12 = 2.0 * det_t * ratio * phase / den

        return s11, s21, s12, s22

    def reflection_transmission_absorption_unequal_refs(
            self,
            Z_in: np.ndarray | complex | float,
            omega: np.ndarray,
            *,
            Z_out: np.ndarray | complex | float | None = None,
            k_ref: np.ndarray | complex | float | None = None,
            length: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return reflection, transmission, and absorption for a 2-port element.

        Parameters
        ----------
        Z_in :
            Reference characteristic impedance at the input port.
        omega :
            Angular frequency array.
        Z_out :
            Reference characteristic impedance at the output port.
            If omitted, the equal-reference case is assumed: Z_out = Z_in.
        k_ref, length :
            Optional phase-reference normalization for transmission.

        Notes
        -----
        Use Z_out != Z_in when inlet and outlet sections differ
        (e.g. cone, expansion, contraction).
        """
        s11, s21, _, _ = self.scattering_coefficients(
            Z_in,
            Z_out,
            omega,
            k_ref=k_ref,
            length=length,
        )
        absorption = 1.0 - np.abs(s11) ** 2 - np.abs(s21) ** 2 *np.real(Z_in)/np.real(Z_out)
        return s11, s21, absorption

    def reflection_transmission_absorption(
        self,
        Z_c: np.ndarray | complex | float,
        omega: np.ndarray,
        *,
        k_ref: np.ndarray | complex | float | None = None,
        length: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return reflection, transmission, and absorption for a 2-port element.

        ``Z_c`` is the reference characteristic impedance used to terminate and
        normalize the transfer matrix. ``k_ref`` and ``length`` are optional and
        only affect the transmission coefficient phase normalization.

        If both ``k_ref`` and ``length`` are provided, the transmission result is
        multiplied by ``exp(-1j * k_ref * length)`` to match finite-cell
        conventions such as those used in ``A0_single_HR.py``.

        If both are omitted, the method returns the raw transfer-matrix
        transmission coefficient without this extra phase-reference factor.
        """
        omega = self._as_omega_array(omega)
        T = self.matrix(omega)

        Z_c = np.asarray(Z_c, dtype=np.complex128)
        if Z_c.ndim == 0:
            Z_c = np.full(T.shape[0], Z_c, dtype=np.complex128)
        Z_c = np.broadcast_to(Z_c, (T.shape[0],))

        A11 = T[:, 0, 0]
        A12 = T[:, 0, 1]
        A21 = T[:, 1, 0]
        A22 = T[:, 1, 1]

        det_t = A11 * A22 - A12 * A21
        den = A11 + A12 / Z_c + A21 * Z_c + A22

        if (k_ref is None) ^ (length is None):
            raise ValueError("k_ref and length must be provided together for phase-normalized transmission")

        phase = 1.0
        if k_ref is not None and length is not None:
            k_ref = np.asarray(k_ref, dtype=np.complex128)
            if k_ref.ndim == 0:
                k_ref = np.full(T.shape[0], k_ref, dtype=np.complex128)
            k_ref = np.broadcast_to(k_ref, (T.shape[0],))
            phase = np.exp(-1j * k_ref * length)

        transmission = 2.0 * phase * det_t / den
        # transmission = 2.0 * phase  / den
        reflection = (A11 + A12 / Z_c - A21 * Z_c - A22) / den
        absorption = 1.0 - np.abs(reflection) ** 2 - np.abs(transmission) ** 2
        return reflection, transmission, absorption

    def bloch_wavenumber(self, omega: np.ndarray, cell_length: float) -> np.ndarray:
        """Return the raw Bloch wavenumber when interpreting the element as one periodic cell."""
        omega = self._as_omega_array(omega)
        if cell_length <= 0.0:
            raise ValueError("cell_length must be positive")
        T = self.matrix(omega)
        trace = T[:, 0, 0] + T[:, 1, 1]
        return np.arccos(trace / 2.0) / cell_length

    def bloch_wavenumber_physical(
        self,
        omega: np.ndarray,
        cell_length: float,
        *,
        enforce_positive_imag: bool = True,
    ) -> np.ndarray:
        """Return a continuity-tracked Bloch branch suitable for plotting and retrieval comparison."""
        q_raw = self.bloch_wavenumber(omega, cell_length)
        q_phys = np.empty_like(q_raw)
        q_phys[0] = q_raw[0]

        two_pi_over_l = 2.0 * np.pi / cell_length
        for idx in range(1, q_raw.size):
            candidates = []
            for sign in (1.0, -1.0):
                base = sign * q_raw[idx]
                for shift in (-1, 0, 1):
                    candidates.append(base + shift * two_pi_over_l)
            q_phys[idx] = min(candidates, key=lambda cand: abs(cand - q_phys[idx - 1]))

        if enforce_positive_imag:
            head = q_phys[: min(10, q_phys.size)]
            if np.mean(np.imag(head)) < 0.0:
                q_phys = -q_phys
        return q_phys

    def retrieve_equivalent_duct(
        self,
        Z_c: np.ndarray | complex | float,
        omega: np.ndarray,
        *,
        k_ref: np.ndarray | complex | float,
        length: float,
        area: float | None = None,
        reflection: np.ndarray | complex | float | None = None,
        transmission: np.ndarray | complex | float | None = None,
        track_branch: bool = False,
    ) -> EquivalentDuctRetrievalResult:
        """Retrieve an equivalent homogeneous duct from a 2-port waveguide system.

        ``Z_c`` and ``k_ref`` define the reference medium surrounding the system,
        and ``length`` is the homogenized thickness used for the retrieval.

        If ``reflection`` and ``transmission`` are not provided, they are computed
        from the system using ``reflection_transmission_absorption``. When a custom
        reflection coefficient is needed, for example a left-side reflection
        instead of the default right-side one, pass it explicitly together with the
        matching transmission coefficient.

        When ``track_branch`` is ``True``, the retrieved ``k_eff`` is continuity-
        tracked across frequency by considering sign flips and ``2*pi/length``
        shifts. This is useful for plotting and comparison to known branches.
        """
        omega = self._as_omega_array(omega)
        if length <= 0.0:
            raise ValueError("length must be positive")

        Z_c = np.asarray(Z_c, dtype=np.complex128)
        if Z_c.ndim == 0:
            Z_c = np.full(omega.shape, Z_c, dtype=np.complex128)
        Z_c = np.broadcast_to(Z_c, (omega.size,))

        k_ref = np.asarray(k_ref, dtype=np.complex128)
        if k_ref.ndim == 0:
            k_ref = np.full(omega.shape, k_ref, dtype=np.complex128)
        k_ref = np.broadcast_to(k_ref, (omega.size,))

        if (reflection is None) ^ (transmission is None):
            raise ValueError("reflection and transmission must be provided together")

        if reflection is None and transmission is None:
            reflection, transmission, _ = self.reflection_transmission_absorption(
                Z_c=Z_c,
                omega=omega,
                k_ref=k_ref,
                length=length,
            )
        else:
            reflection = np.asarray(reflection, dtype=np.complex128)
            transmission = np.asarray(transmission, dtype=np.complex128)
            if reflection.ndim == 0:
                reflection = np.full(omega.shape, reflection, dtype=np.complex128)
            if transmission.ndim == 0:
                transmission = np.full(omega.shape, transmission, dtype=np.complex128)
            reflection = np.broadcast_to(reflection, (omega.size,))
            transmission = np.broadcast_to(transmission, (omega.size,))

        T_prim = transmission * np.exp(1j * k_ref * length)
        Z_eff = Z_c * np.sqrt(
            ((1.0 + reflection) ** 2 - T_prim**2)
            / ((1.0 - reflection) ** 2 - T_prim**2)
        )
        zz = Z_eff / Z_c
        k_eff = (-1.0 / (length * 1j)) * np.log(
            (T_prim * (1.0 - zz)) / (reflection * (1.0 + zz) - zz + 1.0)
        )
        if track_branch and k_eff.size > 1:
            k_eff_tracked = np.empty_like(k_eff)
            k_eff_tracked[0] = k_eff[0]
            two_pi_over_l = 2.0 * np.pi / length
            for idx in range(1, k_eff.size):
                candidates = []
                for sign in (1.0, -1.0):
                    base = sign * k_eff[idx]
                    for shift in (-1, 0, 1):
                        candidates.append(base + shift * two_pi_over_l)
                k_eff_tracked[idx] = min(candidates, key=lambda cand: abs(cand - k_eff_tracked[idx - 1]))
            k_eff = k_eff_tracked

        rho_eff = None
        K_eff = None
        rho_eff_lf = None
        K_eff_lf = None
        if area is not None:
            area = float(area)
            if area <= 0.0:
                raise ValueError("area must be positive when provided")
            rho_eff = - Z_eff * k_eff * area / omega
            K_eff =  - omega * Z_eff / (k_eff * area)
            T = self.matrix(omega)
            T12 = T[:, 0, 1]
            T21 = T[:, 1, 0]
            rho_eff_lf = T12 / (1j * omega * length )*area
            K_eff_lf = 1j * omega * length / (T21 * area)

        from .frozen import EquivalentDuct

        return EquivalentDuctRetrievalResult(
            duct=EquivalentDuct(length=length, k_eff=k_eff, zc_eff=Z_eff),
            Z_eff=Z_eff,
            k_eff=k_eff,
            T_prim=T_prim,
            rho_eff=rho_eff,
            K_eff=K_eff,
            rho_eff_lf=rho_eff_lf,
            K_eff_lf=K_eff_lf,
        )


    @property
    def plot(self):
        """Return a plotting accessor for this waveguide element/system."""
        from ..plotting.waveguide import WaveguidePlotter

        return WaveguidePlotter(self)

    def p_in_from_incident_wave(
        self,
        p0: float | complex,
        Z_load: np.ndarray | complex | float,
        Z_source: np.ndarray | complex | float,
        omega: np.ndarray,
    ) -> np.ndarray:
        """Convert incident-wave amplitude to inlet total pressure.

        ``p0`` is the forward-going plane-wave pressure amplitude in the
        upstream medium. ``Z_source`` is the characteristic/source impedance at
        the inlet plane. The returned value is the total pressure to pass to
        ``p_tm`` for this source convention.
        """
        omega = self._as_omega_array(omega)
        zin = self.Z_in(Z_load, omega)
        source = np.asarray(Z_source, dtype=np.complex128)
        if source.ndim == 0:
            source = np.full(zin.shape, source, dtype=np.complex128)
        source = np.broadcast_to(source, zin.shape)
        return 2.0 * p0 * zin / (source + zin)

    def p_tm(
        self,
        p_in: float | complex,
        Z_load: np.ndarray | complex | float,
        omega: np.ndarray,
    ) -> np.ndarray:
        omega = self._as_omega_array(omega)
        T = self.matrix(omega)
        load = np.asarray(Z_load, dtype=np.complex128)
        if load.ndim == 0:
            load = np.full(T.shape[0], load, dtype=np.complex128)
        load = np.broadcast_to(load, (T.shape[0],))
        p_out = np.empty(T.shape[0], dtype=np.complex128)
        inf_mask = np.isinf(load)
        if np.any(inf_mask):
            p_out[inf_mask] = p_in * 1.0 / T[inf_mask, 0, 0]
        if np.any(~inf_mask):
            p_out[~inf_mask] = p_in * load[~inf_mask] / (T[~inf_mask, 0, 0] * load[~inf_mask] + T[~inf_mask, 0, 1])
        return p_out

    def U_in(
        self,
        p_in: float | complex,
        Z_load: np.ndarray | complex | float,
        omega: np.ndarray,
    ) -> np.ndarray:
        omega = self._as_omega_array(omega)
        zin = self.Z_in(Z_load, omega)
        return np.asarray(p_in, dtype=np.complex128) / zin

    def U_tm(
        self,
        p_in: float | complex,
        Z_load: np.ndarray | complex | float,
        omega: np.ndarray,
    ) -> np.ndarray:
        omega = self._as_omega_array(omega)
        p_out = self.p_tm(p_in, Z_load, omega)
        load = np.asarray(Z_load, dtype=np.complex128)
        if load.ndim == 0:
            load = np.full(p_out.shape, load, dtype=np.complex128)
        load = np.broadcast_to(load, p_out.shape)
        u_out = np.empty_like(p_out)
        inf_mask = np.isinf(load)
        if np.any(inf_mask):
            u_out[inf_mask] = 0.0 + 0.0j
        if np.any(~inf_mask):
            u_out[~inf_mask] = p_out[~inf_mask] / load[~inf_mask]
        return u_out

    def state_in(
        self,
        p_in: float | complex,
        Z_load: np.ndarray | complex | float,
        omega: np.ndarray,
    ) -> np.ndarray:
        omega = self._as_omega_array(omega)
        p_in = np.asarray(p_in, dtype=np.complex128)
        if p_in.ndim == 0:
            p_in = np.full(omega.shape, p_in, dtype=np.complex128)
        p_in = np.broadcast_to(p_in, omega.shape)
        return np.column_stack((p_in, self.U_in(p_in, Z_load, omega)))

    def state_tm(
        self,
        p_in: float | complex,
        Z_load: np.ndarray | complex | float,
        omega: np.ndarray,
    ) -> np.ndarray:
        omega = self._as_omega_array(omega)
        p_out = self.p_tm(p_in, Z_load, omega)
        return np.column_stack((p_out, self.U_tm(p_in, Z_load, omega)))

    def state_in_from_incident_wave(
        self,
        p0: float | complex,
        Z_load: np.ndarray | complex | float,
        Z_source: np.ndarray | complex | float,
        omega: np.ndarray,
    ) -> np.ndarray:
        omega = self._as_omega_array(omega)
        p_in = self.p_in_from_incident_wave(p0, Z_load, Z_source, omega)
        return self.state_in(p_in, Z_load, omega)

    def state_tm_from_incident_wave(
        self,
        p0: float | complex,
        Z_load: np.ndarray | complex | float,
        Z_source: np.ndarray | complex | float,
        omega: np.ndarray,
    ) -> np.ndarray:
        omega = self._as_omega_array(omega)
        p_in = self.p_in_from_incident_wave(p0, Z_load, Z_source, omega)
        return self.state_tm(p_in, Z_load, omega)




class InfiniteLayerModel(AcousticModel, ABC):
    """Model family for infinite-extent barriers/layers, not cascaded ducts."""

    @abstractmethod
    def transmission_coefficient(self, omega: np.ndarray, *, theta: float = 0.0) -> np.ndarray:
        """Complex transmission coefficient for incidence angle ``theta``."""

    @abstractmethod
    def reflection_coefficient(self, ompega: np.ndarray, *, theta: float = 0.0) -> np.ndarray:
        """Complex reflection coefficient for incidence angle ``theta``."""

    def TL(self, omega: np.ndarray, *, theta: float = 0.0) -> np.ndarray:
        omega = self._as_omega_array(omega)
        tau = self.transmission_coefficient(omega, theta=theta)
        return 20.0 * np.log10(1.0 / np.abs(tau))


class SeriesImpedanceElement(WaveguideElement, ABC):
    """Waveguide element defined by an acoustic series impedance in the ``[p, U]`` basis."""

    @abstractmethod
    def acoustic_series_impedance(self, omega: np.ndarray) -> np.ndarray:
        """Return the acoustic series impedance with shape ``(N_freq,)``."""

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        z_series = np.asarray(self.acoustic_series_impedance(omega), dtype=np.complex128).ravel()
        T = np.zeros((z_series.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = 1.0
        T[:, 0, 1] = z_series
        T[:, 1, 1] = 1.0
        return T

class ComposedElement(WaveguideElement):
    """Composition of two waveguide elements by transfer-matrix cascade."""

    def __init__(self, left: WaveguideElement, right: WaveguideElement) -> None:
        self.left = left
        self.right = right

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        L = self.left.matrix(omega)
        R = self.right.matrix(omega)
        return np.einsum("nij,njk->nik", L, R)


class ParallelElement(WaveguideElement):
    """Parallel combination of true 2-port branches sharing the same inlet and outlet nodes.

    The implementation assumes every branch is defined in the ``[p, U]`` basis,
    so that branch admittances can be added directly:

        Y_eq = Y_1 + Y_2 + ...

    where ``Y`` is the 2-port admittance form relating ``[U1, U2]`` to
    ``[p1, p2]``. The equivalent transfer matrix is obtained by converting the
    summed admittance matrix back to the standard ``[p, U]`` transfer form.
    """

    def __init__(self, *branches: WaveguideElement) -> None:
        if len(branches) < 2:
            raise ValueError("ParallelElement requires at least two branches")

        flattened_branches: list[WaveguideElement] = []
        for branch in branches:
            if not isinstance(branch, WaveguideElement):
                raise TypeError(f"Cannot compose object of type {type(branch)!r} in parallel with WaveguideElement")
            if branch.state_basis != "pu":
                raise ValueError("ParallelElement currently requires all branches to use the 'pu' state basis")
            if isinstance(branch, ParallelElement):
                flattened_branches.extend(branch.branches)
            else:
                flattened_branches.append(branch)

        self.branches = tuple(flattened_branches)

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        admittances = [_transfer_to_admittance(branch.matrix(omega)) for branch in self.branches]
        total_admittance = np.add.reduce(admittances)
        return _admittance_to_transfer(total_admittance)


class DecascadedElement(WaveguideElement):
    """Decascade wrapper using left or right matrix removal."""

    def __init__(
        self,
        total: WaveguideElement,
        removed: WaveguideElement,
        side: str = "right",
        *,
        method: Literal["direct", "tikhonov", "lcurve"] = "direct",
        regularization: float | None = None,
        lambda_grid: np.ndarray | None = None,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        if method not in {"direct", "tikhonov", "lcurve"}:
            raise ValueError("method must be 'direct', 'tikhonov', or 'lcurve'")
        if regularization is not None and regularization <= 0.0:
            raise ValueError("regularization must be strictly positive")
        self.total = total
        self.removed = removed
        self.side = side
        self.method = method
        self.regularization = regularization
        if lambda_grid is None:
            lambda_grid = np.logspace(-12.0, 0.0, 121)
        lambda_grid = np.asarray(lambda_grid, dtype=np.float64).ravel()
        if lambda_grid.size == 0:
            raise ValueError("lambda_grid must not be empty")
        if np.any(lambda_grid <= 0.0):
            raise ValueError("lambda_grid must contain strictly positive values")
        self.lambda_grid = lambda_grid

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        total_matrix = self.total.matrix(omega)
        removed_matrix = self.removed.matrix(omega)
        if self.method == "direct":
            recovered = np.empty_like(total_matrix)
            if self.side == "right":
                for idx in range(total_matrix.shape[0]):
                    recovered[idx] = np.linalg.solve(removed_matrix[idx].T, total_matrix[idx].T).T
                return recovered
            for idx in range(total_matrix.shape[0]):
                recovered[idx] = np.linalg.solve(removed_matrix[idx], total_matrix[idx])
            return recovered

        inv_removed_matrix = _invert_frequency_matrices(
            removed_matrix,
            method=self.method,
            regularization=self.regularization,
            lambda_grid=self.lambda_grid,
        )
        if self.side == "right":
            return np.einsum("nij,njk->nik", total_matrix, inv_removed_matrix)
        return np.einsum("nij,njk->nik", inv_removed_matrix, total_matrix)


def _transfer_to_admittance(matrices: np.ndarray) -> np.ndarray:
    t11 = matrices[:, 0, 0]
    t12 = matrices[:, 0, 1]
    t21 = matrices[:, 1, 0]
    t22 = matrices[:, 1, 1]

    if np.any(np.abs(t12) < 1.0e-30):
        raise ValueError("Cannot convert transfer matrix to admittance form when T12 is zero")

    admittance = np.empty_like(matrices)
    admittance[:, 0, 0] = t22 / t12
    admittance[:, 0, 1] = t21 - (t22 * t11) / t12
    admittance[:, 1, 0] = 1.0 / t12
    admittance[:, 1, 1] = -t11 / t12
    return admittance


def _admittance_to_transfer(matrices: np.ndarray) -> np.ndarray:
    y11 = matrices[:, 0, 0]
    y12 = matrices[:, 0, 1]
    y21 = matrices[:, 1, 0]
    y22 = matrices[:, 1, 1]

    if np.any(np.abs(y21) < 1.0e-30):
        raise ValueError("Cannot convert admittance matrix to transfer form when Y21 is zero")

    transfer = np.empty_like(matrices)
    transfer[:, 0, 0] = -y22 / y21
    transfer[:, 0, 1] = 1.0 / y21
    transfer[:, 1, 0] = y12 - (y11 * y22) / y21
    transfer[:, 1, 1] = y11 / y21
    return transfer


def _invert_frequency_matrices(
    matrices: np.ndarray,
    *,
    method: Literal["direct", "tikhonov", "lcurve"],
    regularization: float | None,
    lambda_grid: np.ndarray,
) -> np.ndarray:
    if method == "direct":
        return np.linalg.inv(matrices)
    if method == "tikhonov":
        lambda_value = 1.0e-8 if regularization is None else regularization
        return _tikhonov_inverse_stack(matrices, lambda_value=lambda_value)
    return _lcurve_inverse_stack(matrices, lambda_grid=lambda_grid)


def _tikhonov_inverse_stack(matrices: np.ndarray, *, lambda_value: float) -> np.ndarray:
    inverses = np.empty_like(matrices)
    identity = np.eye(matrices.shape[1], dtype=np.complex128)

    for idx in range(matrices.shape[0]):
        matrix = matrices[idx]
        column_scale = np.linalg.norm(matrix, axis=0)
        column_scale = np.where(column_scale > 1.0e-30, column_scale, 1.0)
        matrix_scaled = matrix / column_scale
        inverse_scaled = np.linalg.solve(matrix_scaled.conj().T @ matrix_scaled + lambda_value * identity, matrix_scaled.conj().T)
        inverses[idx] = inverse_scaled / column_scale[:, np.newaxis]
    return inverses


def _lcurve_inverse_stack(matrices: np.ndarray, *, lambda_grid: np.ndarray) -> np.ndarray:
    inverses = np.empty_like(matrices)
    identity = np.eye(matrices.shape[1], dtype=np.complex128)

    for idx in range(matrices.shape[0]):
        candidates = np.empty((lambda_grid.size, matrices.shape[1], matrices.shape[2]), dtype=np.complex128)
        residual_norms = np.zeros(lambda_grid.size, dtype=np.float64)
        solution_norms = np.zeros(lambda_grid.size, dtype=np.float64)

        matrix = matrices[idx]
        column_scale = np.linalg.norm(matrix, axis=0)
        column_scale = np.where(column_scale > 1.0e-30, column_scale, 1.0)
        matrix_scaled = matrix / column_scale
        gram = matrix_scaled.conj().T @ matrix_scaled
        matrix_h = matrix_scaled.conj().T
        for lambda_idx, lambda_value in enumerate(lambda_grid):
            candidate_scaled = np.linalg.solve(gram + lambda_value * identity, matrix_h)
            candidate = candidate_scaled / column_scale[:, np.newaxis]
            candidates[lambda_idx] = candidate
            residual_norms[lambda_idx] = np.linalg.norm(matrix @ candidate - identity)
            solution_norms[lambda_idx] = np.linalg.norm(candidate)

        best_idx = _lcurve_corner(residual_norms, solution_norms)
        inverses[idx] = candidates[best_idx]
    return inverses


def _lcurve_corner(residuals: np.ndarray, solutions: np.ndarray) -> int:
    x = np.log(np.maximum(residuals, np.finfo(float).tiny))
    y = np.log(np.maximum(solutions, np.finfo(float).tiny))
    if residuals.size < 3:
        return int(np.argmin(residuals))

    curvature = np.zeros(residuals.size, dtype=np.float64)
    for idx in range(1, residuals.size - 1):
        dx = (x[idx + 1] - x[idx - 1]) / 2.0
        dy = (y[idx + 1] - y[idx - 1]) / 2.0
        d2x = x[idx + 1] - 2.0 * x[idx] + x[idx - 1]
        d2y = y[idx + 1] - 2.0 * y[idx] + y[idx - 1]
        denom = (dx**2 + dy**2) ** 1.5 + 1e-30
        curvature[idx] = abs(dx * d2y - dy * d2x) / denom
    return int(np.nanargmax(curvature))


# Compatibility alias kept during the transition from the old name.
AcousticElement = WaveguideElement
