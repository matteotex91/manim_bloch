import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def play_timetrace_3levels(
    omega: float,
    time_vec: np.ndarray,
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
    c_timetrace = np.zeros((3, len(time_vec)), dtype=np.complex_)
    c_timetrace[2, 0] = 1
    for i in range(1, len(time_vec)):
        comp_exp = np.exp(1j * omega * time_vec[i])
        comp_exp = comp_exp + np.conj(comp_exp)
        H_mat = (
            (time_vec[i] - time_vec[i - 1])
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
    omega: float,
    time_vec: np.ndarray,
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
    c = np.complex_([0, 0, 1])
    timetrace = []
    framecount = 0
    for i in tqdm(range(1, len(time_vec))):
        comp_exp = np.exp(1j * omega * time_vec[i])
        comp_exp = comp_exp + np.conj(comp_exp)
        H_mat = (
            (time_vec[i] - time_vec[i - 1])
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
        #  c = c / np.abs(np.linalg.norm(c))
        if framecount % save_every_nframes == 0:
            timetrace.append(c)
            framecount = 0
        framecount += 1
    return np.array(timetrace).T


def concatenate_sequences(detunings, phases, times, Omega_Rs):
    detuning = np.array([])
    phase = np.array([])
    time = np.array([])
    Omega_R = np.array([])
    for d, p, t, o in zip(detunings, phases, times, Omega_Rs):
        detuning = np.append(detuning, d)
        phase = np.append(phase, p)
        time = np.append(time, t + (0 if time.shape[0] == 0 else np.max(time)))
        Omega_R = np.append(Omega_R, o)

    return detuning, phase, time, Omega_R


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
