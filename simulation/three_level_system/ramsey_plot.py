import pickle
import os
import matplotlib.pyplot as plt
import numpy as np


with open(
    os.path.join(os.path.dirname(__file__), "trajectories", "ramsey_1.pickle"), "rb"
) as f:
    datadict = pickle.load(f)

c_timetrace = datadict["c_timetrace"]
time_decimate = datadict["time_decimate"]
phase_decimate = datadict["phase_decimate"]
time = datadict["time"]
omega = datadict["omega"]
Omega_R_eg = datadict["Omega_R_eg"]
Omega_R_fg = datadict["Omega_R_fg"]
omega_e = datadict["omega_e"]
omega_f = datadict["omega_f"]
omega_g = datadict["omega_g"]

plt.plot(
    time_decimate,
    np.abs(c_timetrace[0, :]) / np.abs(np.linalg.norm(c_timetrace, axis=0)),
    label="$c_e$",
)
plt.plot(
    time_decimate,
    np.abs(c_timetrace[1, :]) / np.abs(np.linalg.norm(c_timetrace, axis=0)),
    label="$c_f$",
)
plt.plot(
    time_decimate,
    np.abs(c_timetrace[2, :]) / np.abs(np.linalg.norm(c_timetrace, axis=0)),
    label="$c_g$",
)
plt.legend()
plt.grid(True)
plt.show()


ae = c_timetrace[0, :]
ag = c_timetrace[2, :]
x_vals = np.conj(ag) * ae + ag * np.conj(ae)
y_vals = 1j * (np.conj(ag) * ae - ag * np.conj(ae))
z_vals = ae * np.conj(ae) - ag * np.conj(ag)

plt.plot(time_decimate, np.abs(z_vals), label="abs")
plt.plot(time_decimate, np.angle(z_vals), label="arg")
plt.legend()
plt.grid(True)
plt.show()

counterphase = np.exp(1j * phase_decimate)
ae = c_timetrace[0, :] * counterphase
ag = c_timetrace[2, :] * counterphase
x_vals = np.conj(ag) * ae + ag * np.conj(ae)
y_vals = 1j * (np.conj(ag) * ae - ag * np.conj(ae))
z_vals = ae * np.conj(ae) - ag * np.conj(ag)

plt.plot(time_decimate, np.abs(z_vals), label="abs")
plt.plot(time_decimate, np.angle(z_vals), label="arg")
plt.legend()
plt.grid(True)
plt.show()
