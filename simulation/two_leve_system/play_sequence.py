import numpy as np
import matplotlib.pyplot as plt


def play_timetrace(
    detuning: np.ndarray,
    phase: np.ndarray,
    time: np.ndarray,
    Omega_R: np.ndarray,
) -> np.ndarray:
    R_timetrace = np.zeros((3, len(time)))
    R_timetrace[2, 0] = 1
    T = np.array(
        [
            np.real(Omega_R * np.exp(1j * phase)),
            -np.imag(Omega_R * np.exp(1j * phase)),
            -detuning,
        ]
    )
    for i in range(1, len(time)):
        R_timetrace[:, i] = R_timetrace[:, i - 1] + (time[i] - time[i - 1]) * np.cross(
            T[:, i], R_timetrace[:, i - 1]
        )
        R_timetrace[:, i] = R_timetrace[:, i] / np.linalg.norm(R_timetrace[:, i])
    return R_timetrace


def play_final(
    detuning: np.ndarray,
    phase: np.ndarray,
    time: np.ndarray,
    Omega_R: np.ndarray,
) -> np.ndarray:
    R = np.array([0, 0, 1])
    T = np.array(
        [
            np.real(Omega_R * np.exp(1j * phase)),
            -np.imag(Omega_R * np.exp(1j * phase)),
            -detuning,
        ]
    )
    for i in range(1, len(time)):
        R = R + (time[i] - time[i - 1]) * np.cross(T[:, i], R)
        R = R / np.linalg.norm(R)
    return R


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

    R_timetrace = play_timetrace(detuning, phase, time, Omega_R)
    plt.plot(time, R_timetrace[0, :])
    plt.plot(time, R_timetrace[1, :])
    plt.plot(time, R_timetrace[2, :])
    plt.show()
