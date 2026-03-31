import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONSTANTS
# ============================================================

mu0 = 4*np.pi*1e-7

q_ion = 1.6e-12
q_elec = -1.6e-19

r_obs = np.array([0.5, -0.3, 5.0])

axis = np.array([1.0, 0.3, 0.2])
axis = axis / np.linalg.norm(axis)

times = np.linspace(0, 0.002, 1500)
t0 = 0.0008

ION_LIST = [1000, 10000, 100000]

# ============================================================
# COSINE DISTRIBUTION (TRUE RANDOM)
# ============================================================

def sample_cosine_direction(axis):

    axis = axis / np.linalg.norm(axis)

    u = np.random.rand()
    phi = 2*np.pi*np.random.rand()

    cos_theta = np.sqrt(u)   # inverse transform
    sin_theta = np.sqrt(1 - cos_theta**2)

    if abs(axis[0]) < 0.9:
        e1 = np.cross(axis, [1,0,0])
    else:
        e1 = np.cross(axis, [0,1,0])

    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    return (
        cos_theta*axis +
        sin_theta*(np.cos(phi)*e1 + np.sin(phi)*e2)
    )

# ============================================================
# VELOCITY
# ============================================================

def sample_velocity(mean, sigma):
    return max(np.random.normal(mean, sigma), 0)

# ============================================================
# RANDOM POSITION (NEW — VERY IMPORTANT)
# ============================================================

def sample_position(radius=0.05):
    # small disk around origin
    r = radius * np.sqrt(np.random.rand())
    theta = 2*np.pi*np.random.rand()

    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = np.random.randn()*0.01   # small vertical spread

    return np.array([x, y, z])

# ============================================================
# PULSE (NO NORMALIZATION)
# ============================================================

def impact_pulse(t, t0):

    tau_rise = 5e-6
    tau_decay = 2e-4

    y = np.zeros_like(t)
    mask = t >= t0
    tt = t[mask] - t0

    y[mask] = np.exp(-tt/tau_decay) - np.exp(-tt/tau_rise)

    return y

# ============================================================
# BIOT–SAVART
# ============================================================

def biot_savart(q, v_vec, r_vec):

    R = np.linalg.norm(r_vec)

    if R < 1e-9:
        return np.zeros(3)

    return (mu0/(4*np.pi)) * q * np.cross(v_vec, r_vec) / (R**3)

# ============================================================
# SIMULATION
# ============================================================

def simulate_plasma(N):

    Bx = np.zeros_like(times)
    By = np.zeros_like(times)
    Bz = np.zeros_like(times)

    for _ in range(N):

        # random origin (NEW!)
        r0 = sample_position()

        # ION
        dir_i = sample_cosine_direction(axis)
        v_i = sample_velocity(10000, 3000) * dir_i

        r_vec_i = r_obs - r0
        dB_i = biot_savart(q_ion, v_i, r_vec_i)

        # ELECTRON
        dir_e = sample_cosine_direction(axis)
        v_e = sample_velocity(15000, 4000) * dir_e

        r_vec_e = r_obs - r0
        dB_e = biot_savart(q_elec, v_e, r_vec_e)

        # time randomness
        t_shift = t0 + np.random.randn()*2e-5

        pulse = impact_pulse(times, t_shift)

        Bx += (dB_i[0] + dB_e[0]) * pulse
        By += (dB_i[1] + dB_e[1]) * pulse
        Bz += (dB_i[2] + dB_e[2]) * pulse

    return Bx, By, Bz

# ============================================================
# RUN
# ============================================================

for N in ION_LIST:

    print(f"\nRunning N = {N}")

    Bx, By, Bz = simulate_plasma(N)
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    plt.figure(figsize=(10,6))

    plt.plot(times, Bx, label="Bx")
    plt.plot(times, By, label="By")
    plt.plot(times, Bz, label="Bz")
    plt.plot(times, Bmag, label="|B|", linewidth=2)

    # find actual peak time from magnitude
    peak_idx = np.argmax(Bmag)
    t_peak = times[peak_idx]

    plt.axvline(t_peak, linestyle="--", color="black",
                label=f"Peak time = {t_peak:.6f} s")
    plt.title(f"Monte Carlo Plasma Field — {N} particles")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnetic Field (T)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
