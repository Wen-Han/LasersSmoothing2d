import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# focal length of the final lens, in meter
focal_length = 7.7
# diameter of the whole laser beam, in meter
beam_aperture = 0.35
# laser wave length, in meter
wave_length = 0.351e-9
# number of beamlets
n_beams = [64, 64]
# type of smoothing. valid options are 'FM SSD', 'RPM SSD', 'ISI'
lsType = 'FM SSD'
# number of color cycles
ncc = [12.2, 12.2]
# (RMS value of) the amplitude of phase modulation
beta = 4
if beta <= 0.0:
    # static speckle pattern. all models are reduced to phase plate
    nuTotal = 0.0
    nu = 0.0
else:
    # bandwidth of the optics, normalized to laser frequency
    nuTotal = 0.001 / np.sqrt(2)
    # bandwidth of phase modulation, normalized to laser frequency
    nu = nuTotal / 2 / beta
# electric field amplitude of each beamlet, scalar or 1d numpy array
e0 = 1.0
# complex transform for each beamlet, scalar or 1d numpy array
epsilon_n = 1.0
# length of the movie, normalized to 1/laser frequency. (# of laser cycle)
tMaxMovie = 1e4
# length of the time series, normalized to 1/laser frequency. (# of laser cycle)
tMax = 2e2 + tMaxMovie
# time delay imposed by one echelon step in ISI, in 1/nuTotal
tDelay = 1.5
# delta time for the movie, in 1/laser frequency, the code will round it so as
# to complete tMax with integer steps. Increasing dt can reduce calculation
# time. Does not apply to FM SSD.
dt = 100.0
# ------------------------------------------------------------------------------

# XDL unit
xdl = wave_length / beam_aperture


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
phi_n = np.pi * np.random.binomial(1, 0.5, (n_beams[0], n_beams[1]))
# x0, x1 are normalized to the beam aperture
x0, x1 = np.meshgrid(np.linspace(-0.5, 0.5, num=n_beams[0]),
                     np.linspace(-0.5, 0.5, num=n_beams[1]))


def ssd_2d_fm(t):
    """ Beamlets after SSD and before the final focal len (2d version).

    :param t: current time
    :return: near field electric field amplitude of the full beam
    """
    psi_n = beta * (np.sin(2 * np.pi * nu * (t + s[0] * x0)) +
                    np.sin(2 * np.pi * nu * (t + s[1] * x1)))
    return general_form_beamlets_2d(e0, epsilon_n, psi_n, phi_n)


# s is the parameter for gratings in SSD. equal to time delay in xdl units
if nu > 0:
    s = np.divide(ncc, nu)
else:
    s = [0.0, 0.0]


def ssd_2d_rpm_init():
    """ initialize the random phase modulation sequence

    :return: time series of random phases in two directions
    """
    # make sure tMax is larger than the time delay introduced by gratings
    global tMax
    global s
    if nu > 0:
        tMax += np.max(np.divide(ncc, nu))
        s = np.divide(ncc, nu)
    return gen_gaussian_time_series(np.long(tMax / dt), nu * 2, beta)


tn_d = np.arange(0.0, n_beams[0] * n_beams[1]).reshape(n_beams)
tn = np.long(0)


def isi_2d_init():
    """ initialize isi related parameters

    :return: time series of the phase
    """
    global tMax
    global tn_d
    if nuTotal > 0:
        tMax += np.prod(n_beams) / nuTotal * tDelay
        tn_d *= tDelay / nuTotal / dt
        tn_d = tn_d.astype(long)
    # this may take a while ...
    random_phase = gen_gaussian_time_series(np.long(tMax / dt / 2),
                                            nuTotal / np.pi, np.pi)
    return np.reshape(random_phase, random_phase.size)


def gen_gaussian_time_series(t_num, fwhm, rms_mean):
    """ generate a time series that has gaussian power spectrum

    :param t_num: number of grid points in time
    :param fwhm: full width half maximum of the power spectrum
    :param rms_mean: root-mean-square average of the spectrum
    :return: a time series array with shape [2, t_num]
    """
    global tn
    global dt
    tn = t_num
    dt = tMax / tn
    if fwhm == 0.0:
        return np.zeros((2, tn))
    pm_phase = np.random.uniform(-0.5, 0.5, (2, tn))
    spec = np.fft.fftshift(np.fft.fft(pm_phase), axes=-1)
    omega = np.fft.fftshift(np.fft.fftfreq(t_num, d=dt))
    spec_filter = np.exp(- np.square(omega / fwhm * 2 *
                                     np.sqrt(0.5 * np.log(2))))
    spec[0, :] *= spec_filter
    spec[1, :] *= spec_filter
    time_seq_filtered = np.real(np.fft.ifft(np.fft.ifftshift(spec)))
    rms = np.sqrt(np.mean(np.square(time_seq_filtered), axis=1))
    pm_phase[0, :] = time_seq_filtered[0, :] / rms[0] * rms_mean
    pm_phase[1, :] = time_seq_filtered[1, :] / rms[1] * rms_mean
    return pm_phase


def ssd_2d_rpm(t):
    """ Beamlets after SSD and before the final focal len (2d version).

    :param t: current time
    :return: near field electric field amplitude of the full beam
    """
    # dt = tMax / tn
    indx = np.array([(t - s[0] * x0.flatten()) / dt,
                     (t - s[1] * x1.flatten()) / dt])
    indx = indx.astype(long)
    psi_n = np.reshape(pmPhase[0, indx[0, :]] + pmPhase[1, indx[1, :]],
                       n_beams)
    return general_form_beamlets_2d(e0, epsilon_n, psi_n, phi_n)


def isi_2d(t):
    """ Beamlets after ISI and before the final focal len (2d version).

    :param t: current time
    :return: near field electric field amplitude of the full beam
    """
    tt = np.long(t / dt)
    indx = np.array(tt - tn_d.flatten())
    indx = np.mod(indx.astype(long), tn)
    psi_n = np.reshape(pmPhase[indx], n_beams)
    return general_form_beamlets_2d(e0, epsilon_n, psi_n, phi_n)


func_dict = {
    'FM SSD': ssd_2d_fm,
    'RPM SSD': ssd_2d_rpm,
    'ISI': isi_2d,
}


def select_laser_smoothing_2d(pm_type='FM SSD'):
    """ select which smoothing technique to use

    :param pm_type: a string denoting the type, defaulting to 'FM SSD'
    :return: ssd function, ssd_2d_fm or ssd_2d_rpm
    """
    if pm_type.upper() == 'RPM SSD':
        pm_phase = ssd_2d_rpm_init()
    elif pm_type.upper() == 'ISI':
        pm_phase = isi_2d_init()
    else:
        pm_type = 'FM SSD'
        pm_phase = None
    return func_dict[pm_type.upper()], pm_phase


def focal_len_2d(beamlets):
    """ Use the diffraction integral to calculate the interference of beamlets on focal plane (2d version).

    :param beamlets: electric field of full beam
    :return: far fields pattern on the focal plane
    """
    field = np.multiply(proPhase, np.fft.fftshift(np.fft.fft2(beamlets)))
    return field

time = 0
laser_smoothing_2d, pmPhase = select_laser_smoothing_2d(lsType)

axis_color = 'lightgoldenrodyellow'
fig = plt.figure()
fig.canvas.set_window_title(lsType)

fig.add_subplot(121)
# fig.subplots_adjust(left=0.1, bottom=0.25)
bl = laser_smoothing_2d(time)
nn = bl.shape
im0 = plt.imshow(np.real(bl), cmap='gray', aspect='equal',
                 extent=[-0.5, 0.5, -0.5, 0.5])
# im0 = plt.imshow(np.square(np.real(bl)), cmap='gray', aspect='equal')
plt.title('E before final focal lens')
plt.xlabel('x\' ($D$)')
plt.ylabel('y\' ($D$)')
# bl_tmp = bl
gn = bl.shape
xfp0, xfp1 = np.meshgrid(np.linspace(-0.5 * gn[0], 0.5 * gn[0], num=gn[0]),
                         np.linspace(-0.5 * gn[1], 0.5 * gn[1], num=gn[1]))
# constant phase shift due to beam propagation
proPhase = np.exp(1j * (np.square(xfp0) + np.square(xfp1))
                  * xdl * focal_length / beam_aperture * np.pi +
                  2j * np.pi * focal_length / wave_length)

ax = fig.add_subplot(122)
fp_tmp = focal_len_2d(bl)
im1 = plt.imshow(np.square(np.real(fp_tmp)), cmap='gray', aspect='equal',
                 extent=[-n_beams[0] / 2 + 0.5, n_beams[0] / 2 - 0.5,
                         -n_beams[1] / 2 + 0.5, n_beams[1] / 2 - 0.5])
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
    im1.set_data(np.square(np.real(tmp)))
    fig.canvas.draw_idle()
time_slider.on_changed(sliders_on_changed)


plt.show()
