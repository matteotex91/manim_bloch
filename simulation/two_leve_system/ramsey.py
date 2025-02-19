import numpy as np
import matplotlib.pyplot as plt
from play_sequence import play_timetrace, concatenate_sequences, play_final


t_sim = 1e-6
Omega_R_val = 6.23e6  # rad/s
omega_eg = 2836.8e6

splitting = 74e6  # Hz

pitime = 512e-9
pitime_2 = pitime / 2

fe_time = 200e-9

# pulse 1
ndiv_1 = 1000
time_1 = np.linspace(0, pitime_2, ndiv_1)
phase_1 = np.zeros(ndiv_1)
detuning_1 = np.zeros(ndiv_1)
Omega_R_1 = np.full(ndiv_1, Omega_R_val)

# free evolution
ndiv_2 = 5000
time_2 = np.linspace(0, fe_time, ndiv_2)
phase_2 = np.zeros(ndiv_2)
detuning_2 = np.full(ndiv_2, splitting / 2)
Omega_R_2 = np.zeros(ndiv_2)


# pulse 2
ndiv_3 = 1000
time_3 = np.linspace(0, pitime_2, ndiv_3)
phase_3 = np.zeros(ndiv_3)
detuning_3 = np.zeros(ndiv_3)
Omega_R_3 = np.full(ndiv_3, Omega_R_val)

detuning, phase, time, Omega_R = concatenate_sequences(
    [detuning_1, detuning_2, detuning_3],
    [phase_1, phase_2, phase_3],
    [time_1, time_2, time_3],
    [Omega_R_1, Omega_R_2, Omega_R_3],
)
R_t = play_timetrace(detuning, phase, time, Omega_R)
plt.plot(time, R_t[0, :])
plt.plot(time, R_t[1, :])
plt.plot(time, R_t[2, :])
plt.show()
