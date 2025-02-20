import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from play_sequence import play_timetrace_3levels_decimate, concatenate_sequences


D0 = 2.87e9
splitting = 74e6  # Hz

omega_e = D0 + splitting / 2
omega_f = D0 - splitting / 2
omega_g = 0
omega = omega_e  # drive frequency
pitime_1 = 160e-9
pitime_2 = 200e-9

Omega_R_eg = np.pi / pitime_1
Omega_R_fg = np.pi / pitime_2

### Time settings
dt = 1e-13
pitime_ndiv = int(pitime_1 / dt)

tau_fe = 250e-9
tau_fe_ndiv = int(tau_fe / dt)


time_1 = np.linspace(0, pitime_1, pitime_ndiv)
omega_1 = np.full(pitime_ndiv, omega)
Omega_R_eg_1 = np.full(pitime_ndiv, Omega_R_eg)
Omega_R_fg_1 = np.full(pitime_ndiv, Omega_R_fg)


time_2 = np.linspace(0, tau_fe, tau_fe_ndiv)
omega_2 = np.full(tau_fe_ndiv, omega)
Omega_R_eg_2 = np.full(tau_fe_ndiv, 0)
Omega_R_fg_2 = np.full(tau_fe_ndiv, 0)

time_3 = np.linspace(0, pitime_1, pitime_ndiv)
omega_3 = np.full(pitime_ndiv, omega)
Omega_R_eg_3 = np.full(pitime_ndiv, Omega_R_eg)
Omega_R_fg_3 = np.full(pitime_ndiv, Omega_R_fg)

time, omega, Omega_R_eg, Omega_R_fg = concatenate_sequences(
    [time_1, time_2, time_3],
    [omega_1, omega_2, omega_3],
    [Omega_R_eg_1, Omega_R_eg_2, Omega_R_eg_3],
    [Omega_R_fg_1, Omega_R_fg_2, Omega_R_fg_3],
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
