"""Plotting helpers for waveguide elements and composed systems."""

from __future__ import annotations

import numpy as np


class WaveguidePlotter:
    """Matplotlib plotting accessor for waveguide elements."""

    def __init__(self, element) -> None:
        self.element = element

    def bloch_wavenumber(
        self,
        *,
        omega: np.ndarray | None = None,
        freq: np.ndarray | None = None,
        q: np.ndarray | None = None,
        cell_length: float,
        k0: np.ndarray | None = None,
        fig=None,
        axs=None,
    ):
        """Plot Bloch phase and attenuation branches for one periodic cell.

        Provide either ``omega`` so ``q`` is computed from the element, or
        provide both ``freq`` and precomputed ``q``.
        """
        if cell_length <= 0.0:
            raise ValueError("cell_length must be positive")
        if q is None and omega is None:
            raise ValueError("omega must be provided when q is not provided")
        if q is not None and freq is None:
            raise ValueError("freq must be provided when q is provided")
        if q is None:
            omega = np.asarray(omega, dtype=np.float64).ravel()
            freq = omega / (2.0 * np.pi)
            q = self.element.bloch_wavenumber(omega, cell_length=cell_length)
        else:
            q = np.asarray(q, dtype=np.complex128).ravel()
            freq = np.asarray(freq, dtype=np.float64).ravel()
        if fig is None or axs is None:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        ax_phase, ax_att = axs
        ax_phase.plot(np.real(q) * cell_length / np.pi, freq, label=r'Re($q$)L/$\pi$')
        ax_phase.plot(-np.real(q) * cell_length / np.pi, freq, label=r'-Re($q$)L/$\pi$')
        if k0 is not None:
            ax_phase.plot(k0 * cell_length / np.pi, freq, '--', color='k', label=r'$k_0L/\pi$')
        ax_phase.set_xlim(-1, 1)
        ax_phase.set_ylim(freq[0], freq[-1])
        ax_phase.set_xlabel('Phase constant')
        ax_phase.set_ylabel('Frequency [Hz]')
        ax_phase.grid(True)
        ax_phase.legend()

        ax_att.plot(np.imag(q) * cell_length / np.pi, freq, label=r'Im($q$)L/$\pi$')
        ax_att.plot(-np.imag(q) * cell_length / np.pi, freq, label=r'-Im($q$)L/$\pi$')
        ax_att.set_ylim(freq[0], freq[-1])
        ax_att.set_xlabel('Attenuation constant')
        ax_att.set_ylabel('Frequency [Hz]')
        ax_att.grid(True)
        ax_att.legend()
        fig.tight_layout()
        return fig, axs

    def rta(self, *, freq: np.ndarray, R: np.ndarray, T: np.ndarray, A: np.ndarray, fig=None, axs=None):
        """Plot reflection, transmission, and absorption versus frequency."""
        freq = np.asarray(freq, dtype=np.float64).ravel()
        R = np.asarray(R)
        T = np.asarray(T)
        A = np.asarray(A)
        if fig is None or axs is None:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        ax_rt, ax_a = axs
        ax_rt.plot(freq, np.abs(T), label='|T|')
        ax_rt.plot(freq, np.abs(R), label='|R|')
        ax_rt.set_ylabel('Magnitude')
        ax_rt.set_ylim(-0.05, 1.05)
        ax_rt.grid(True)
        ax_rt.legend()

        ax_a.plot(freq, A, label=r'$1-|R|^2-|T|^2$')
        ax_a.set_xlabel('Frequency [Hz]')
        ax_a.set_ylabel('Absorption')
        ax_a.set_ylim(-0.05, 1.05)
        ax_a.grid(True)
        ax_a.legend()
        fig.tight_layout()
        return fig, axs
