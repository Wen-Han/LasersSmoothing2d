import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv USER INPUT BEGINS vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv #
# /                                                                             \ #
#                                                                                 #
# ======================== General parameters ==================== #
# number of beamlets along each transverse direction
n_beamlets = np.array([32, 32], dtype='int')
# number of grid points in each transverse direction
n_grid = np.array([64, 64], dtype='int')
# F number
F = 10.0
# types of smoothing. valid options are:
# 'FM SSD', 'GS RPM SSD', 'GS ISI'
lsType = 'GS RPM SSD'
# bandwidth of the optics, normalized to laser frequency
laser_bandwidth = 0.005
# length of the movie, normalized to 1/f0.
tmax = 4e3

# ======================== SSD parameters ========================= #
# Only support single FM for now
# the amplitude of phase along each direction
phase_mod_amp = (4.1, 4.1)
# number of color cycles
ncc = [1.4, 1.0]
# bandwidth distributed with respect to the two transverse direction
ssd_distr = [1.2, 1]
#                                                                               #
# \                                                                           / #
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ USER INPUT ENDS ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ #

# ================== Sanity checks on user inputs ===================== #
assert laser_bandwidth > 0, 'laser_bandwidth must be greater than 0'
for q in (n_beamlets, phase_mod_amp, n_grid, ncc, ssd_distr):
    assert np.size(q) == 2, 'has to be a size 2 array'
for q in (ncc, ssd_distr, phase_mod_amp):
    assert q[0] > 0 or q[1] > 0, 'cannot be all zeros'
supported_bandwidth = 'FM SSD', 'GS RPM SSD', 'GS ISI'
assert lsType.upper() in supported_bandwidth, 'Only support one of the following: ' + ', '.join(supported_bandwidth)

# ================== Calculate auxiliary variables ================== #
if 'SSD' in lsType.upper():
    phase_plate = np.random.uniform(-np.pi, np.pi, size=n_beamlets[0]*n_beamlets[1]).reshape(n_beamlets)
elif 'ISI' in lsType.upper():
    phase_plate = np.zeros(n_beamlets)  # ISI does not require phase plates
else:
    raise NotImplementedError

L = 1.0 * n_beamlets  # transverse size, in XDL units (λf/D=λF)
ssd_frac = np.sqrt(ssd_distr[0]**2 + ssd_distr[1]**2)
ssd_frac = ssd_distr[0] / ssd_frac, ssd_distr[1] / ssd_frac
phase_mod_freq = [laser_bandwidth * sf * 0.5 / pma for sf, pma in zip(ssd_frac, phase_mod_amp)]
n = [np.linspace(-0.5*(n_beamlets[0]-1), 0.5*(n_beamlets[0]-1), num=n_beamlets[0]),
     np.linspace(-0.5*(n_beamlets[1]-1), 0.5*(n_beamlets[1]-1), num=n_beamlets[1])]
xn1, xn0 = np.meshgrid(n[1], n[0])
xp1, xp0 = np.meshgrid(np.linspace(0, L[1], num=n_grid[1], endpoint=False),
                       np.linspace(0, L[0], num=n_grid[0], endpoint=False))
nn1, nn0 = np.meshgrid(n[1]+0.5*(n_beamlets[1]-1), n[0]+0.5*(n_beamlets[0]-1))
phase_mod_phase = np.random.standard_normal(2) * np.pi
td = (ncc[0] / phase_mod_freq[0] if phase_mod_freq[0] > 0 else 0,
      ncc[1] / phase_mod_freq[1] if phase_mod_freq[1] > 0 else 0)
# time interval to update the speckle pattern, roughly update 50 time every bandwidth cycle
tu = 1 / laser_bandwidth / 50
time = np.arange(0, tmax, tu)


# ======================= Initialization ========================= #
def gen_gaussian_time_series(t_num, fwhm, rms_mean):
    """ generate a time series that has gaussian power spectrum

    :param t_num: number of grid points in time
    :param fwhm: full width half maximum of the power spectrum
    :param rms_mean: root-mean-square average of the spectrum
    :return: a time series array of complex numbers with shape [t_num]
    """
    if fwhm == 0.0:
        return np.zeros((2, t_num))
    omega = np.fft.fftshift(np.fft.fftfreq(t_num, d=tu))
    # rand_ph = np.random.normal(scale=np.pi, size=t_num)
    psd = np.exp(-np.log(2) * 0.5 * np.square(omega / fwhm * 2 * np.pi))
    psd *= np.sqrt(t_num) / np.sqrt(np.mean(np.square(psd))) * rms_mean
    pm_phase = np.array(psd) * (np.random.normal(size=t_num) +
                                1j * np.random.normal(size=t_num))
    pm_phase = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(pm_phase)))
    pm_phase *= rms_mean / np.sqrt(np.mean(np.square(np.abs(pm_phase))))
    return pm_phase


def init_GS_timeseries():
    if 'SSD' in lsType.upper():
        pm_phase0 = gen_gaussian_time_series(time.size + int(np.sum(td) / tu) + 2,
                                             2 * np.pi * phase_mod_freq[0], phase_mod_amp[0])
        pm_phase1 = gen_gaussian_time_series(time.size + int(np.sum(td) / tu) + 2,
                                             2 * np.pi * phase_mod_freq[1], phase_mod_amp[1])
        time_interp = np.arange(start=0, stop=time[-1]+np.sum(td)+3*tu, step=tu)[:pm_phase0.size]
        return (time_interp,
                [(np.real(pm_phase0) + np.imag(pm_phase0)) / np.sqrt(2),
                 (np.real(pm_phase1) + np.imag(pm_phase1)) / np.sqrt(2)])
    elif 'ISI' in lsType.upper():
        complex_amp = np.stack([np.stack([gen_gaussian_time_series(time.size, 2 * laser_bandwidth, 1)
                                          for _i in range(n_beamlets[1])])
                                for _j in range(n_beamlets[0])])
        return time, complex_amp


if 'GS' in lsType.upper():
    time_ext, timeSeries = init_GS_timeseries()
else:
    time_ext, timeSeries = time, None


def beamlets_complex_amplitude(t, lsType='FM SSD'):
    if lsType.upper() == 'FM SSD':
        phase_t = (phase_mod_amp[0] * np.sin(phase_mod_phase[0] +
                                             2 * np.pi * phase_mod_freq[0] * (t - xn0 * td[0] / n_beamlets[0])) +
                   phase_mod_amp[1] * np.sin(phase_mod_phase[1] +
                                             2 * np.pi * phase_mod_freq[1] * (t - xn1 * td[1] / n_beamlets[1])))
        return np.exp(1j * phase_t)
    elif lsType.upper() == 'GS RPM SSD':
        phase_t = (np.interp(t + nn0 * td[0] / n_beamlets[0], time_ext, timeSeries[0]) +
                   np.interp(t + nn1 * td[1] / n_beamlets[1], time_ext, timeSeries[1]))
        return np.exp(1j * phase_t)
    elif lsType.upper() == 'GS ISI':
        return timeSeries[:, :, int(round(t/tu))]
    else:
        raise NotImplementedError


static_speckles = np.zeros((n_beamlets[0], n_beamlets[1], n_grid[0], n_grid[1]), dtype='complex128')
for i in range(n_beamlets[0]):
    for j in range(n_beamlets[1]):
        static_speckles[i, j, ...] = np.exp(1j * (phase_plate[i, j] -
                                                  (n[0][i] * xp0 / n_beamlets[0] +
                                                   n[1][j] * xp1 / n_beamlets[1]) * 2 * np.pi))


def generate_speckle_pattern(tm):
    tnow = tm * tu
    bca = beamlets_complex_amplitude(tnow, lsType=lsType)
    # combining the static speckle pattern with the changing beamlets
    speckle_amp = np.sum(bca[:, :, np.newaxis, np.newaxis] * static_speckles, axis=(0, 1))

    # plot and save
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 5.3))
    fig.canvas.manager.set_window_title(lsType)

    im0 = ax[0].imshow(np.fmod(np.angle(bca) + phase_plate * 0, np.pi),
                       cmap='bwr', aspect='equal', extent=[0, n_beamlets[0], 0, n_beamlets[1]],
                       vmin=-np.pi, vmax=np.pi, origin='lower'
                       )
    ax[0].set_title('Near field beamlet phases\n(excluding phase plate)')
    ax[0].set_xlabel('beamlet number along x')
    ax[0].set_ylabel('beamlet number along y')
    fig.colorbar(im0, ax=ax[0], location="top")

    im1 = ax[1].imshow(np.abs(speckle_amp), cmap='gray', aspect='equal', extent=[0, L[0], 0, L[1]],
                       interpolation='sinc', vmin=0, vmax=np.sqrt(9 * n_beamlets[0] * n_beamlets[1]),
                       origin='lower'
                       )
    ax[1].set_title('Far field speckle patterns')
    ax[1].set_xlabel(r'$x~ (\lambda F)$')
    ax[1].set_ylabel(r'$y~ (\lambda F)$')
    fig.colorbar(im1, ax=ax[1], location="top")
    plt.suptitle(lsType + r', t=%.2f $\delta f^{-1}$' % (tnow * laser_bandwidth))
    plt.savefig('test' + "{0:0>4}".format(tm) + '.png')
    plt.close()


# for t_m in range(time.size):
#     if t_m % (time.size // 10) == 0:
#         print(t_m // (time.size // 10) * 10, '% done')
#     generate_speckle_pattern(t_m)
def init_worker(data0, data1, data2, data3):
    global static_speckles
    global time_ext
    global timeSeries
    global phase_mod_phase
    static_speckles, time_ext, timeSeries, phase_mod_phase = data0, data1, data2, data3


if __name__ == '__main__':
    with Pool(initializer=init_worker, initargs=(static_speckles, time_ext, timeSeries, phase_mod_phase)) as pool:
        pool.map(generate_speckle_pattern, range(time.size))
