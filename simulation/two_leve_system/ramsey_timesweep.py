import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from play_sequence import concatenate_sequences, play_final
from tqdm import tqdm
import scipy


br = []
fe_times = np.linspace(4e-9, 1000e-9, 100)


t_sim = 1e-6

omega_eg = 2836.8e6

pitime = 512e-9
pitime_2 = pitime / 2
Omega_R_val = 2 * np.pi / (2 * pitime)

splitting = 74e6  # Hz


gamma_NV = 28e9
# detuning_free = (
#     -(splitting) / (2 * gamma_NV) * (scipy.constants.e / (2 * scipy.constants.m_e))
# )
detuning_free = splitting / (2 * 2)
print(detuning_free)

# actual period in the experiment : 820 ns
# simulated : 170 ns
# ratio : 4.8 time


# pulse 1
ndiv_1 = 200
time_1 = np.linspace(0, pitime_2, ndiv_1)
phase_1 = np.zeros(ndiv_1)
detuning_1 = np.zeros(ndiv_1)
Omega_R_1 = np.full(ndiv_1, Omega_R_val)

# pulse 2
ndiv_3 = 200
time_3 = np.linspace(0, pitime_2, ndiv_3)
phase_3 = np.zeros(ndiv_3)
detuning_3 = np.zeros(ndiv_3)
Omega_R_3 = np.full(ndiv_3, Omega_R_val)


for fe_time in tqdm(fe_times):
    # free evolution
    ndiv_2 = 500
    time_2 = np.linspace(0, fe_time, ndiv_2)
    phase_2 = np.zeros(ndiv_2)
    detuning_2 = np.full(ndiv_2, detuning_free)
    Omega_R_2 = np.zeros(ndiv_2)

    detuning, phase, time, Omega_R = concatenate_sequences(
        [detuning_1, detuning_2, detuning_3],
        [phase_1, phase_2, phase_3],
        [time_1, time_2, time_3],
        [Omega_R_1, Omega_R_2, Omega_R_3],
    )
    R_t = play_final(detuning, phase, time, Omega_R)
    br.append(R_t[2])
plt.plot(fe_times, br)
plt.show()
