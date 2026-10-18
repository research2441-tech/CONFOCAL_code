"""System parameters shared by every module (single source of truth)."""
import numpy as np

C0 = 299792458.0

class Cfg:
    # --- carrier, band, array ---
    fc = 300e9                 # carrier frequency (Hz)
    B = 30e9                   # bandwidth (Hz)
    M = 32                     # representative subcarriers
    N = 512                    # BS antennas (ULA)
    K = 32                     # subarrays, one TTD each
    lam = C0 / fc
    d = lam / 2                # half-wavelength spacing at fc

    # --- hardware ---
    tau_max = 0.40e-9          # TTD range (s)
    b_tau = 5                  # TTD resolution bits
    b_phi = 4                  # phase-shifter bits
    P_ps = 20e-3               # W per phase shifter
    P_ttd = 60e-3              # W per TTD
    P_rf = 200e-3              # W RF chain
    P_bb = 300e-3              # W baseband
    P_tx = 1.0                 # W radiated power
    eta_pa = 0.3               # PA efficiency

    # --- link budget ---
    snr_tx_db = 92.0           # P_tx G_t G_r/(N0 B) in dB, path loss excluded
    kappa = 0.0005             # molecular absorption coefficient (1/m) at fc
    nlos_paths = 2
    block_att_db = 25.0        # attenuation of LoS on blocked antennas

    # --- frame / overhead ---
    T_f = 0.5e-3               # frame (beam coherence block) duration (s)
    t_p = 0.5e-6               # dwell per beam probe incl. reconfiguration and report
    H = 8                      # forecast horizon (frames)
    Q_reacq = 420              # probes used by sector re-acquisition

    # --- scenario ---
    u_lim = 0.80               # sector |sin theta| <= u_lim
    r_min, r_max = 3.0, 30.0

    @classmethod
    def freqs(cls):
        m = np.arange(cls.M)
        return cls.fc + cls.B * ((m + 0.5) / cls.M - 0.5)

    @classmethod
    def ant_pos(cls):
        delta = np.arange(cls.N) - (cls.N - 1) / 2
        return delta * cls.d

    @classmethod
    def slots(cls):
        return cls.T_f / cls.t_p

    @classmethod
    def delta_u(cls):
        return 1.0 / cls.N * 2.0 * 0.5   # half of null-to-null width (2/N)

    @classmethod
    def delta_rho(cls, u):
        beta2 = 1.2 ** 2
        return 2 * cls.lam * beta2 / ((cls.N * cls.d) ** 2 * np.maximum(1 - u ** 2, 0.2))
