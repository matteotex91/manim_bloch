import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def play_timetrace_3levels(
    time_vec: np.ndarray,
    omega: float,
    Omega_R_eg: np.ndarray,
    Omega_R_fg: np.ndarray,
    omega_e: float,
    omega_f: float,
    omega_g: float,
) -> np.ndarray:
    """
    c[0,:] = timetrace of c_e
    c[1,:] = timetrace of c_f
    c[2,:] = timetrace of c_g
    Initial state : c[:,0]=[0,0,1], ground state
    """
    c_timetrace = np.zeros((3, len(time_vec)), dtype=np.complex128)
    c_timetrace[2, 0] = 1
    phase = 0
    for i in range(1, len(time_vec)):
        dt = time_vec[i] - time_vec[i - 1]
        phase += dt * omega[i]
        comp_exp = np.exp(1j * phase)
        comp_exp = comp_exp + np.conj(comp_exp)
        H_mat = (
            dt
            * (-0.5j)
            * np.array(
                [
                    [2 * omega_e, 0, comp_exp * Omega_R_eg[i]],
                    [0, 2 * omega_f, comp_exp * Omega_R_fg[i]],
                    [
                        comp_exp * np.conj(Omega_R_eg[i]),
                        comp_exp * np.conj(Omega_R_fg[i]),
                        2 * omega_g,
                    ],
                ]
            )
        )
        c_timetrace[:, i] = c_timetrace[:, i - 1] + np.dot(H_mat, c_timetrace[:, i - 1])
        c_timetrace[:, i] = c_timetrace[:, i] / np.linalg.norm(c_timetrace[:, i])
    return c_timetrace


def play_timetrace_3levels_decimate(
    time_vec: np.ndarray,
    omega: np.ndarray,
    Omega_R_eg: np.ndarray,
    Omega_R_fg: np.ndarray,
    omega_e: float,
    omega_f: float,
    omega_g: float,
    save_every_nframes: int,
) -> np.ndarray:
    """
    c[0,:] = timetrace of c_e
    c[1,:] = timetrace of c_f
    c[2,:] = timetrace of c_g
    Initial state : c[:,0]=[0,0,1], ground state
    """
    c = np.complex128([1e-2, 1e-2, 1])
    timetrace = []
    framecount = 0
    phase = 0
    time_decimate = []
    phase_decimate = []
    for i in tqdm(range(1, len(time_vec))):
        dt = time_vec[i] - time_vec[i - 1]
        phase += dt * omega[i]
        comp_exp = np.exp(1j * phase)
        comp_exp = comp_exp + np.conj(comp_exp)
        H_mat = (
            dt
            * (-0.5j)
            * np.array(
                [
                    [2 * omega_e, 0, comp_exp * Omega_R_eg[i]],
                    [0, 2 * omega_f, comp_exp * Omega_R_fg[i]],
                    [
                        comp_exp * np.conj(Omega_R_eg[i]),
                        comp_exp * np.conj(Omega_R_fg[i]),
                        2 * omega_g,
                    ],
                ]
            )
        )
        c = c + np.dot(H_mat, c)
        c = c / np.abs(np.linalg.norm(c))
        if framecount % save_every_nframes == 0:
            timetrace.append(c)
            time_decimate.append(time_vec[i])
            phase_decimate.append(phase)
        framecount += 1
    return np.array(time_decimate), np.array(phase_decimate), np.array(timetrace).T


def concatenate_sequences(
    time_vecs: np.ndarray,
    omegas: np.array,
    Omega_R_egs: np.ndarray,
    Omega_R_fgs: np.ndarray,
):
    time = np.array([])
    omega = np.array([])
    Omega_R_eg = np.array([])
    Omega_R_fg = np.array([])
    for t, o, o1, o2 in zip(time_vecs, omegas, Omega_R_egs, Omega_R_fgs):
        time = np.append(time, t + (0 if time.shape[0] == 0 else np.max(time)))
        omega = np.append(omega, o)
        Omega_R_eg = np.append(Omega_R_eg, o1)
        Omega_R_fg = np.append(Omega_R_fg, o2)

    return time, omega, Omega_R_fg, Omega_R_fg


if __name__ == "__main__":
    ndiv = 1000
    t_sim = 1e-6
    Omega_R = 6.23e6  # rad/s
    omega_eg = 2836.8e6

    detuning = np.zeros(ndiv)
    time = np.linspace(0, t_sim, ndiv)
    phase = np.zeros(ndiv)

    R_timetrace = play_timetrace_3levels(detuning, phase, time, Omega_R)
    plt.plot(time, R_timetrace[0, :])
    plt.plot(time, R_timetrace[1, :])
    plt.plot(time, R_timetrace[2, :])
    plt.show()
