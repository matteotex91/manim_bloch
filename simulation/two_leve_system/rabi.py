import numpy as np
import matplotlib.pyplot as plt

### RF settings
# Rabi frequency
Omega_R = 6.23e6  # rad/s
# Splitting
splitting = 74e6  # Hz
# drive frequency
omega = 2836.8e6
# transition frequency
omega_eg = 2836.8e6
# detuning (omega-omega_eg)
delta = omega - omega_eg

### Initial state : bright [0,0,1]
# dark instead is [0,0,-1]
R = np.array([0, 0, 1])

### Time settings
t_sim = 1e-6
dt = 1e-9
t_ndiv = int(t_sim / dt)


# effective rabi frequency
Omega = np.sqrt(np.square(np.abs(Omega_R)) + np.square(delta))

# Rabi vector
T = np.array([np.real(Omega_R), -np.imag(Omega_R), -delta])

# Computation
R_timetrace = np.zeros((3, t_ndiv))
R_timetrace[:, 0] = R
for i in range(1, t_ndiv):
    R_timetrace[:, i] = R_timetrace[:, i - 1] + dt * np.cross(T, R_timetrace[:, i - 1])

plt.plot(R_timetrace[0, :])
plt.plot(R_timetrace[1, :])
plt.plot(R_timetrace[2, :])
plt.show()
print("stop here")
