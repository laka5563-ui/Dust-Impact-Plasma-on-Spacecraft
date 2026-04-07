import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from stl import mesh as stl_mesh

# ============================================================
# CONSTANTS
# ============================================================

mu0 = 4*np.pi*1e-7
q_ion = 1.6e-19

# ============================================================
# HELIOCENTRIC DISTANCE (PSP)
# ============================================================

R_sun_radius = 6.96e8
R_psp_surface = 6.1e9
R_psp = R_psp_surface + R_sun_radius
R_ref = 1.496e11

# ============================================================
# TIME 
# ============================================================

times = np.linspace(1.7, 1.8, 1200)
dt = times[1] - times[0]
t0_true = 1.747

# ============================================================
# IMPACT PULSE
# ============================================================

def impact_pulse(t, t0):
    y = np.zeros_like(t)
    tt = t - t0
    mask = tt >= 0

    rise = 1 - np.exp(-tt[mask]/6e-4)
    decay = np.exp(-tt[mask]/3e-3)

    y[mask] = rise * decay
    return y

# ============================================================
# SIGNAL 
# ============================================================

def generate_measured_data():

    pulse = impact_pulse(times, t0_true)

    Bx = -2.2e-10 * pulse
    By =  1.8e-10 * pulse
    Bz = -1.3e-10 * pulse

    Bx += -1.8e-10 * np.exp(-((times - t0_true)/0.0005)**2)
    By +=  1.2e-10 * np.exp(-((times - (t0_true+0.0002))/0.0007)**2)
    Bz += -0.8e-10 * np.exp(-((times - (t0_true-0.0001))/0.0006)**2)

    decay = np.exp(-(times - t0_true)/0.008)
    decay[times < t0_true] = 0

    Bx += -0.5e-10 * decay
    By +=  0.3e-10 * decay
    Bz += -0.2e-10 * decay

    osc = 0.015e-9 * np.sin(600*(times - t0_true)) * np.exp(-(times-t0_true)/0.01)
    osc[times < t0_true] = 0

    Bx += osc
    By += 0.8 * osc
    Bz += 0.6 * osc

    noise = 0.01e-9
    Bx += noise*np.random.randn(len(times))
    By += noise*np.random.randn(len(times))
    Bz += noise*np.random.randn(len(times))

    return Bx, By, Bz

# ============================================================
# GEOMETRY
# ============================================================

def load_stl_geometry(path):
    m = stl_mesh.Mesh.from_file(path)
    verts = m.vectors.reshape(-1, 3)
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    m.vectors -= center
    return m

# ============================================================
# INVERSE SOLVER
# ============================================================

def estimate_impact_inverse(mesh, SCM, B_peak):

    best_error = np.inf
    best_point = None
    best_tri = -1

    for i, tri in enumerate(mesh.vectors):

        p = (tri[0] + tri[1] + tri[2]) / 3
        r = SCM - p
        r_norm = np.linalg.norm(r)

        if r_norm < 1e-6:
            continue

        B_model = r / (r_norm**3)

        scale = np.dot(B_peak, B_model) / np.dot(B_model, B_model)
        B_model_scaled = scale * B_model

        error = np.linalg.norm(B_peak - B_model_scaled)

        if error < best_error:
            best_error = error
            best_point = p
            best_tri = i

    return best_point, best_tri, best_error

# ============================================================
# PHYSICS
# ============================================================

def estimate_ions(Bmag, distance):

    peak_idx = np.argmax(Bmag)
    window = 30
    B_local = Bmag[peak_idx-window:peak_idx+window]

    alpha0 = 2e-6

    plasma_scale = (R_ref / R_psp)**2
    velocity_scale = (R_ref / R_psp)**0.5
    velocity_effect = velocity_scale**3
    dust_scale = (R_ref / R_psp)**1.3

    raw_scale = plasma_scale * velocity_effect * dust_scale

    gamma = 0.28
    effective_scale = raw_scale**gamma

    alpha = alpha0 * effective_scale

    I = alpha * (2*np.pi*distance*B_local)/mu0
    Q_total = np.sum(I) * dt

    electron_loss_factor = 0.3
    Q_effective = Q_total * (1 - electron_loss_factor)

    ions = Q_effective / q_ion

    return ions

# ============================================================
# VALIDATION FUNCTIONS
# ============================================================

def forward_model_error(point, SCM, B_peak):
    r = SCM - point
    r_norm = np.linalg.norm(r)
    B_model = r / (r_norm**3)

    scale = np.dot(B_peak, B_model) / np.dot(B_model, B_model)
    B_model_scaled = scale * B_model

    return np.linalg.norm(B_peak - B_model_scaled)


def get_top_solutions(mesh, SCM, B_peak, top_n=5):
    solutions = []

    for i, tri in enumerate(mesh.vectors):
        p = (tri[0] + tri[1] + tri[2]) / 3
        err = forward_model_error(p, SCM, B_peak)
        solutions.append((err, i, p))

    solutions.sort(key=lambda x: x[0])
    return solutions[:top_n]


def sensitivity_test(mesh, SCM, B_peak, noise_level=0.05):
    noise = noise_level * B_peak * np.random.randn(3)
    B_perturbed = B_peak + noise

    new_point, _, _ = estimate_impact_inverse(mesh, SCM, B_perturbed)
    return new_point

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    STL_PATH = r"C:/Users/klpra/OneDrive/Desktop/CU Boulder/Academics/Spring 2026/IS/PSP_CAD/PSP_CAD/PSP_Simplified.stl"

    mesh = load_stl_geometry(STL_PATH)
    SCM = np.array([-0.3, -0.5, 4.3])

    Bx, By, Bz = generate_measured_data()

    Bx = gaussian_filter1d(Bx, 1)
    By = gaussian_filter1d(By, 1)
    Bz = gaussian_filter1d(Bz, 1)

    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    peak_idx = np.argmax(Bmag)
    peak_time = times[peak_idx]

    B_peak = np.array([
        Bx[peak_idx],
        By[peak_idx],
        Bz[peak_idx]
    ])

    # INVERSE
    impact_point, tri_idx, error = estimate_impact_inverse(mesh, SCM, B_peak)
    distance = np.linalg.norm(SCM - impact_point)
    ions = estimate_ions(Bmag, distance)

    # FORWARD CHECK
    forward_err = forward_model_error(impact_point, SCM, B_peak)

    print("\n===== FORWARD MODEL CHECK =====")
    print("Difference:", forward_err)

    # TOP SOLUTIONS
    print("\n===== TOP 5 SOLUTIONS =====")

    top_solutions = get_top_solutions(mesh, SCM, B_peak)

    for err, tri, pt in top_solutions:
        print(f"Error: {err:.3e}, Triangle: {tri}, Point: {pt}")

    # SENSITIVITY TEST
    print("\n===== SENSITIVITY TEST =====")

    new_point = sensitivity_test(mesh, SCM, B_peak)

    print("Original impact:", impact_point)
    print("New impact:", new_point)

    # FINAL OUTPUT
    print("\n===== IMPACT ESTIMATION =====")
    print("Triangle:", tri_idx)
    print("Coordinates:", impact_point)
    print("Distance:", distance)
    print("Residual:", error)
    print("Ions:", ions)

# ================= PLOT =================

plt.figure(figsize=(12,4))

plt.plot(times, Bx*1e9, color='blue', label="Bx")
plt.plot(times, By*1e9, color='orange', label="By")
plt.plot(times, Bz*1e9, color='green', label="Bz")
plt.plot(times, Bmag*1e9, color='red', linewidth=2, label="|B|")

plt.axvline(peak_time, linestyle='--', color='black')

plt.xlim(peak_time - 0.01, peak_time + 0.02)

plt.legend()
plt.grid(alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("SCM (nT)")

plt.show()
