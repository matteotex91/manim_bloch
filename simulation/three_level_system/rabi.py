import numpy as np
import matplotlib.pyplot as plt

from play_sequence import play_timetrace_3levels_decimate

D0 = 2.87e9
splitting = 74e6  # Hz

omega_e = D0 + splitting / 2
omega_f = D0 - splitting / 2
omega_g = 0
omega = omega_e  # drive frequency

Omega_R = 6.23e6

### Time settings
t_sim = 1e-6
dt = 1e-12
t_ndiv = int(t_sim / dt)


# Computation
t_decimate, c_timetrace = play_timetrace_3levels_decimate(
    np.linspace(0, t_sim, t_ndiv),
    np.full(t_ndiv, omega),
    np.full(t_ndiv, Omega_R),
    np.full(t_ndiv, Omega_R),
    omega_e,
    omega_f,
    omega_g,
    250,
)

plt.plot(
    t_decimate,
    np.abs(c_timetrace[0, :]) / np.abs(np.linalg.norm(c_timetrace, axis=0)),
    label="$c_e$",
)
plt.plot(
    t_decimate,
    np.abs(c_timetrace[1, :]) / np.abs(np.linalg.norm(c_timetrace, axis=0)),
    label="$c_f$",
)
plt.plot(
    t_decimate,
    np.abs(c_timetrace[2, :]) / np.abs(np.linalg.norm(c_timetrace, axis=0)),
    label="$c_g$",
)
plt.legend()
plt.grid(True)
plt.show()
print("stop here")
