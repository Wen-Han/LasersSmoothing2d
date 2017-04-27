import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from multiprocessing import Pool

# focal length of the final lens, in meter
focal_length = 7.7
# diameter of the whole laser beam, in meter
beam_aperture = 0.35
# laser wave length, in meter
wave_length = 0.351e-9
# number of beamlets
n_beams = [64, 64]
# grid points in each direction
n_grid = [128, 128]
# types of smoothing. valid options are:
# 'FM SSD', 'GS RPM SSD', 'AR RPM SSD', 'GS ISI', 'AR ISI'
lsType = 'AR RPM SSD'
# if apply simple average to AR(1)
if_sma = True
# number of color cycles
ncc = [1.0, 1.0]
# (RMS value of) the amplitude of phase modulation
beta = 4
# bandwidth of the optics, normalized to laser frequency
nuTotal = 0.00117
# electric field amplitude of each beamlet, scalar or 1d numpy array
e0 = 1.0
# complex transform for each beamlet, scalar or 1d numpy array
epsilon_n = 1.0
# length of the movie, normalized to 1/omega0.
tMaxMovie = 2.2e3
# time delay imposed by one echelon step in ISI, in 1/nuTotal
tDelay = 1.5
# delta time for the movie, in 1/omega_0, the code will round it so as
# to complete tMax with integer steps. Increasing dt can reduce calculation
# time. Does not apply to FM SSD in interactive plot
dt = 200.0
# interactive plot or saving to files
interactive_plot = False
# ------------------------------------------------------------------------------
# input ends

# XDL unit
xdl = wave_length / beam_aperture

ncc = np.array(ncc)
lsType = lsType.upper()
# length of the time series, normalized to 1/omega0.
tMax = dt + tMaxMovie

if 'SSD' in lsType:
    nuTotal /= np.sqrt(2)
    if beta > 0:
        nu = 0.5 * nuTotal / beta
    else:
        nu = 0
    # s is the parameter for gratings in SSD. equal to time delay in xdl units
    if nu > 0:
        s = np.divide(2 * np.pi * ncc, nu)
    else:
        s = [0.0, 0.0]


def general_form_beamlets_2d(amp, trans_n, psi_n, ps_n):
    """ General form of the beamlets (1d version).

    E0(x,t)=amp \sum\limits_n e^{i \psi_n} \trans_n \exp[i \ps_n]
    :param amp: field amplitude of the beamlets
    :param trans_n: quantities that define complex transformation for beamlets
    :param psi_n: describe the phase and temporal bandwidth of each beamlet
    :param ps_n: phase shift of each beamlet due to phase plate
    :return: full beam consist of all beamlets
    """
    beamlets = amp * trans_n * np.exp(1j * (psi_n + ps_n))
    return beamlets


# phase plate doesn't change over time
# RPP
# phi_n = np.pi * np.random.binomial(1, 0.5, (n_beams[0], n_beams[1]))
# CPP
phi_n = np.pi * np.random.uniform(-np.pi, np.pi, (n_beams[0], n_beams[1]))
# x0, x1 are normalized to the beam aperture
x0, x1 = np.meshgrid(np.linspace(-0.5, 0.5, num=n_beams[0]),
                     np.linspace(-0.5, 0.5, num=n_beams[1]))


def sma1d(pha, num):
    ret = np.cumsum(pha)
    ret[num:] = ret[num:] - ret[:-num]
    return ret / num


def ar1(b, sigma, pha, num=1):
    """ Autoregressive process of order 1 
    """
    return b * pha + np.random.normal(scale=sigma, size=num)


def ssd_2d_fm(t):
    """ Beamlets after SSD and before the final focal len (2d version).

    :param t: current time
    :return: near field electric field amplitude of the full beam
    """
    psi_n = beta * (np.sin(nu * (t + s[0] * x0)) +
                    np.sin(nu * (t + s[1] * x1)))
    return general_form_beamlets_2d(e0, epsilon_n, psi_n, phi_n)

# time delay array for beamlets
tn_d = np.arange(0.0, n_beams[0] * n_beams[1]).reshape(n_beams)
tn = np.long(0)


def sma_ar1(ttn, tot_bandwidth, pm_am):
    n = 0
    if if_sma:
        # generate a longer series to avoid edge effect of SMA
        n = int(1 / (tot_bandwidth / pm_am ** 2) / dt)
        ttn += 2 * n
    # AR1 phase
    random_phase = np.zeros(ttn)
    # PSD of AR1 process is Lorentzian
    pm_bw = 0.5 * tot_bandwidth / (pm_am * pm_am)
    arcoeff1 = np.exp(- dt * pm_bw)
    arcoeff2 = np.sqrt(1 - arcoeff1 * arcoeff1) * pm_am
    # Discard the first part of the random sequence
    for ti in range(256):
        random_phase[0] = ar1(arcoeff1, arcoeff2, random_phase[0])
    for ti in range(1, ttn):
        random_phase[ti] = ar1(arcoeff1, arcoeff2, random_phase[ti - 1])
    if if_sma:
        denor = (arcoeff1 * arcoeff1 * (n * arcoeff1 - n + 2) -
                 2 * np.power(arcoeff1, n + 1) * (
                     arcoeff1 - 1) + n - arcoeff1 * (n + 2))
        nor = ((1 - arcoeff1 * arcoeff1) * np.square(1 - arcoeff1) * n * n)
        var = np.sqrt(nor / denor)
        random_phase = sma1d(random_phase, n) * var
    return random_phase[n:ttn - n]


def ssd_2d_rpm_init():
    """ initialize the random phase modulation sequence

    :return: time series of random phases in two directions
    """
    # make sure tMax is larger than the time delay introduced by gratings
    global tMax
    global tn_d
    global tn
    global dt
    if nu > 0:
        tMax += np.sum(s)
        # s = np.divide(ncc, nu)
    tn = np.long(tMax / dt)
    dt = tMax / tn
    tdx0, tdx1 = np.meshgrid(np.linspace(0.0, 1.0, num=n_beams[0]),
                             np.linspace(0.0, 1.0, num=n_beams[1]))
    tn_d = np.zeros((2, n_beams[0], n_beams[1]))
    tn_d[0, :, :] = tdx0 * s[0] / dt
    tn_d[1, :, :] = tdx1 * s[1] / dt
    tn_d = tn_d.astype(long)
    # random phases in x and y direction are independent
    ph = np.zeros((2, tn))
    if 'GS' in lsType:
        ph[0, :] = gen_gaussian_time_series(tn, 0.5 * nuTotal / beta, beta)
        ph[1, :] = gen_gaussian_time_series(tn, 0.5 * nuTotal / beta, beta)
    else:
        ph[0, :] = sma_ar1(tn, nuTotal, beta)
        ph[1, :] = sma_ar1(tn, nuTotal, beta)
    return ph


def isi_2d_init():
    """ initialize isi related parameters

    :return: time series of the phase
    """
    global tMax
    global tn_d
    global tn
    global dt
    # make sure we generate a long enough phase sequence
    if nuTotal > 0:
        tMax += (np.prod(n_beams) + 1) / nuTotal * tDelay
        tn_d *= tDelay / nuTotal / dt
        tn_d = tn_d.astype(long)
    tn = np.long(tMax / dt)
    dt = tMax / tn
    # this may take a while ...
    if 'GS' in lsType:
        random_phase = gen_gaussian_time_series(tn,
                                                0.5 * nuTotal / beta, beta)
    else:
        random_phase = sma_ar1(tn, nuTotal, beta)
    return np.reshape(random_phase, random_phase.size)


def gen_gaussian_time_series(t_num, fwhm, rms_mean):
    """ generate a time series that has gaussian power spectrum

    :param t_num: number of grid points in time
    :param fwhm: full width half maximum of the power spectrum
    :param rms_mean: root-mean-square average of the spectrum
    :return: a time series array with shape [t_num]
    """
    if fwhm == 0.0:
        return np.zeros((2, t_num))
    omega = np.fft.fftshift(np.fft.fftfreq(t_num, d=dt))
    rand_ph = np.random.normal(scale=np.pi, size=t_num)
    psd = np.exp(-np.log(2) * 0.5 * np.square(omega / fwhm * 2 * np.pi))
    psd *= np.sqrt(2 * t_num) / np.sqrt(np.mean(np.square(psd))) * rms_mean
    pm_phase = np.array(psd) * (np.cos(rand_ph) + 1j * np.sin(rand_ph))
    pm_phase = np.real(np.fft.ifft(np.fft.ifftshift(pm_phase)))
    return pm_phase


def ssd_2d_rpm(t):
    """ Beamlets after SSD and before the final focal len (2d version).

    :param t: current time
    :return: near field electric field amplitude of the full beam
    """
    # dt = tMax / tn
    # indx = np.array([(t - s[0] * x0.flatten()) / dt,
    #                  (t - s[1] * x1.flatten()) / dt])
    tt = np.long(t / dt)
    indx0 = tt + tn_d[0, :, :].flatten()
    indx1 = tt + tn_d[1, :, :].flatten()
    psi_n = np.reshape(pmPhase[0, indx0] + pmPhase[1, indx1],
                       n_beams)
    return general_form_beamlets_2d(e0, epsilon_n, psi_n, phi_n)


def isi_2d(t):
    """ Beamlets after ISI and before the final focal len (2d version).

    :param t: current time
    :return: near field electric field amplitude of the full beam
    """
    tt = np.long(t / dt)
    indx = np.array(tt + tn_d.flatten())
    psi_n = np.reshape(pmPhase[indx], n_beams)
    return general_form_beamlets_2d(e0, epsilon_n, psi_n, phi_n)


func_dict = {
    'FM SSD': ssd_2d_fm,
    'GS RPM SSD': ssd_2d_rpm,
    'AR RPM SSD': ssd_2d_rpm,
    'AR ISI': isi_2d,
    'GS ISI': isi_2d,
}


def select_laser_smoothing_2d(pm='FM SSD'):
    """ select which smoothing technique to use

    :param pm: a string denoting the type, defaulting to 'FM SSD'
    :return: ssd function, ssd_2d_fm or ssd_2d_rpm
    """
    if 'RPM SSD' in pm:
        pm_phase = ssd_2d_rpm_init()
    elif 'ISI' in pm:
        pm_phase = isi_2d_init()
    else:
        pm = 'FM SSD'
        # No random phase needed for FM SSD
        pm_phase = None
    return func_dict[pm], pm_phase

dx = np.divide(2 * np.pi, n_grid)
xlp0, xlp1 = np.meshgrid(np.linspace(-0.5 * n_beams[0], 0.5 * n_beams[0],
                                     num=n_beams[0]),
                         np.linspace(-0.5 * n_beams[1], 0.5 * n_beams[1],
                                     num=n_beams[1]))


def focal_len_2d(beamlets):
    """ Use the diffraction integral to calculate the interference of beamlets on focal plane (2d version).

    :param beamlets: electric field of full beam
    :return: far fields pattern on the focal plane
    """
    field = np.zeros(n_grid, dtype=complex)
    if n_beams[0] == n_grid[0] and n_beams[1] == n_grid[1]:
        field = np.fft.fft2(beamlets)
    else:
        # naive sum to calculate the Fourier transform
        for ibx in range(-n_grid[0] / 2, n_grid[0] - n_grid[0] / 2):
            for iby in range(-n_grid[1] / 2, n_grid[1] - n_grid[1] / 2):
                field[ibx, iby] = np.sum(np.multiply(
                    np.exp(1j * (ibx * dx[0] * xlp0 + iby * dx[1] * xlp1)),
                    beamlets))
    field = np.multiply(proPhase, np.fft.fftshift(field))
    return field


def plot_2d_xy(fld, tm):
    """ save the absolute values of fld as a png file
    """
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(np.abs(np.real(fld)), cmap='gray', aspect='equal',
               extent=[-n_beams[0] / 2 + 0.5, n_beams[0] / 2 - 0.5,
                       -n_beams[1] / 2 + 0.5, n_beams[1] / 2 - 0.5],
               interpolation=inter_opt, vmin=0,
               vmax=np.sqrt(9 * n_beams[0] * n_beams[1]))
    plt.title('$E_{env}$ (' + lsType.upper() +
              ', t='+"{0:.3f}".format(tm*dt*1.86e-4)+'ps)')
    plt.xlabel('$x (\lambda f/D)$')
    plt.ylabel('$y (\lambda f/D)$')
    plt.savefig('test'+"{0:0>4}".format(tm)+'.png')
    plt.close()

time = 0
laser_smoothing_2d, pmPhase = select_laser_smoothing_2d(lsType)

gn = n_grid
xfp0, xfp1 = np.meshgrid(np.linspace(-0.5 * gn[0], 0.5 * gn[0], num=gn[0]),
                         np.linspace(-0.5 * gn[1], 0.5 * gn[1], num=gn[1]))
# constant phase shift due to beam propagation
proPhase = np.exp(1j * (np.square(xfp0) + np.square(xfp1))
                  * xdl * focal_length / beam_aperture * np.pi +
                  2j * np.pi * focal_length / wave_length)
# setting up some plotting options
inter_opt = 'sinc'

if interactive_plot:
    bl = laser_smoothing_2d(time)
    fp_tmp = focal_len_2d(bl)
    axis_color = 'lightgoldenrodyellow'
    fig = plt.figure()
    fig.canvas.set_window_title(lsType)

    fig.add_subplot(121)
    im0 = plt.imshow(np.real(bl), cmap='gray', aspect='equal',
                     extent=[-0.5, 0.5, -0.5, 0.5])
    plt.title('E before final focal lens')
    plt.xlabel('x\' ($D$)')
    plt.ylabel('y\' ($D$)')
    ax = fig.add_subplot(122)
    im1 = plt.imshow(np.abs(np.real(fp_tmp)), cmap='gray', aspect='equal',
                     extent=[-n_beams[0] / 2 + 0.5, n_beams[0] / 2 - 0.5,
                             -n_beams[1] / 2 + 0.5, n_beams[1] / 2 - 0.5],
                     interpolation=inter_opt, vmin=0,
                     vmax=np.sqrt(9 * n_beams[0] * n_beams[1]))
    plt.title('Intensity near focal spot')
    plt.xlabel('$x (\lambda f/D)$')
    plt.ylabel('$y (\lambda f/D)$')
    fig.set_tight_layout(True)
    # fig.subplots_adjust(left=0.1, bottom=0.25)

    # Add sliders for tweaking the parameters
    time_slider_ax = fig.add_axes([0.2, 0.05, 0.65, 0.03], axisbg=axis_color)
    time_slider = Slider(time_slider_ax, 'Time (laser cycle)',
                         0.0, tMaxMovie, valfmt='%1d', valinit=time)

    def sliders_on_changed(val):
        bl_new = laser_smoothing_2d(time_slider.val)
        im0.set_data(np.real(bl_new))
        # im0.set_data(np.square(np.real(bl_new)))
        tmp = focal_len_2d(bl_new)
        im1.set_data(np.abs(np.real(tmp)))
        fig.canvas.draw_idle()

    time_slider.on_changed(sliders_on_changed)
    plt.show()
else:

    def speckle_time_t(i):
        speckle = laser_smoothing_2d(i * dt)
        fp_speckle = focal_len_2d(speckle)
        plot_2d_xy(fp_speckle, i)

    tnMaxMovie = int(tMaxMovie / dt)
    # the phase information have been known/constructed for any time t, each
    # process can calculate different frames simultaneously/independently
    if __name__ == '__main__':
        pool = Pool()
        pool.map(speckle_time_t, range(tnMaxMovie))
