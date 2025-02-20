import numpy as np
import pickle
import os
from tqdm import tqdm


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


D0 = 2.87e9
splitting = 74e6  # Hz

omega_e = D0 + splitting / 2
omega_f = D0 - splitting / 2
omega_g = 0
pitime_1 = 160e-9
pitime_2 = 200e-9
pitime_1_half = pitime_1 / 2
pitime_2_half = pitime_2 / 2

Omega_R_eg = np.pi / pitime_1
Omega_R_fg = np.pi / pitime_2

### Time settings
dt = 1e-13
pitime_1_ndiv = int(pitime_1 / dt)
pitime_2_ndiv = int(pitime_2 / dt)
pitime_1_half_ndiv = int(pitime_1_half / dt)
pitime_2_half_ndiv = int(pitime_2_half / dt)

tau_fe = 250e-9
tau_fe_ndiv = int(tau_fe / dt)

# pi half on m0 (e,1)
time_1 = np.linspace(0, pitime_1, pitime_1_half_ndiv)
omega_1 = np.full(pitime_1_half_ndiv, omega_e)
Omega_R_eg_1 = np.full(pitime_1_half_ndiv, Omega_R_eg)
Omega_R_fg_1 = np.full(pitime_1_half_ndiv, Omega_R_fg)

# fe
time_2 = np.linspace(0, tau_fe, tau_fe_ndiv)
omega_2 = np.full(tau_fe_ndiv, omega_e)
Omega_R_eg_2 = np.full(tau_fe_ndiv, 0)
Omega_R_fg_2 = np.full(tau_fe_ndiv, 0)

# pi on m0 (e,1)
time_3 = np.linspace(0, pitime_1, pitime_1_ndiv)
omega_3 = np.full(pitime_1_ndiv, omega_e)
Omega_R_eg_3 = np.full(pitime_1_ndiv, Omega_R_eg)
Omega_R_fg_3 = np.full(pitime_1_ndiv, Omega_R_fg)

# pi on p0 (f,2)
time_4 = np.linspace(0, pitime_2, pitime_2_ndiv)
omega_4 = np.full(pitime_2_ndiv, omega_f)
Omega_R_eg_4 = np.full(pitime_2_ndiv, Omega_R_eg)
Omega_R_fg_4 = np.full(pitime_2_ndiv, Omega_R_fg)

# pi on m0 (e,1)
time_5 = np.linspace(0, pitime_1, pitime_1_ndiv)
omega_5 = np.full(pitime_1_ndiv, omega_e)
Omega_R_eg_5 = np.full(pitime_1_ndiv, Omega_R_eg)
Omega_R_fg_5 = np.full(pitime_1_ndiv, Omega_R_fg)

# fe
time_6 = np.linspace(0, tau_fe, tau_fe_ndiv)
omega_6 = np.full(tau_fe_ndiv, omega_e)
Omega_R_eg_6 = np.full(tau_fe_ndiv, 0)
Omega_R_fg_6 = np.full(tau_fe_ndiv, 0)


# pi half on p0 (f,2)
time_7 = np.linspace(0, pitime_2, pitime_2_half_ndiv)
omega_7 = np.full(pitime_2_half_ndiv, omega_f)
Omega_R_eg_7 = np.full(pitime_2_half_ndiv, Omega_R_eg)
Omega_R_fg_7 = np.full(pitime_2_half_ndiv, Omega_R_fg)

time, omega, Omega_R_eg, Omega_R_fg = concatenate_sequences(
    [time_1, time_2, time_3, time_4, time_5, time_6, time_7],
    [omega_1, omega_2, omega_3, omega_4, omega_5, omega_6, omega_7],
    [
        Omega_R_eg_1,
        Omega_R_eg_2,
        Omega_R_eg_3,
        Omega_R_eg_4,
        Omega_R_eg_5,
        Omega_R_eg_6,
        Omega_R_eg_7,
    ],
    [
        Omega_R_fg_1,
        Omega_R_fg_2,
        Omega_R_fg_3,
        Omega_R_fg_4,
        Omega_R_fg_5,
        Omega_R_fg_6,
        Omega_R_fg_7,
    ],
)

time_decimate, phase_decimate, c_timetrace = play_timetrace_3levels_decimate(
    time,
    omega,
    Omega_R_eg,
    Omega_R_fg,
    omega_e,
    omega_f,
    omega_g,
    250,
)
datadict = {
    "c_timetrace": c_timetrace,
    "time_decimate": time_decimate,
    "phase_decimate": phase_decimate,
    "time": time,
    "omega": omega,
    "Omega_R_eg": Omega_R_eg,
    "Omega_R_fg": Omega_R_fg,
    "omega_e": omega_e,
    "omega_f": omega_f,
    "omega_g": omega_g,
}

with open(
    os.path.join(os.path.dirname(__file__), "trajectories", "ramsey_1.pickle"), "wb"
) as f:
    pickle.dump(datadict, f)
