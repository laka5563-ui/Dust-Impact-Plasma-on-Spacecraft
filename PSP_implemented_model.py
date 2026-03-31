import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from stl import mesh as stl_mesh

# ============================================================
# USER SETTINGS
# ============================================================

STL_PATH = r"C:/Users/klpra/OneDrive/Desktop/CU Boulder/Academics/Spring 2026/IS/PSP_CAD/PSP_CAD/PSP_Simplified.stl"

mu0 = 4*np.pi*1e-7
q_ion = 1.6e-12

ION_COUNTS = [1000, 10000, 100000]

# REAL SENSOR POSITION 
SCM_OFFSET = np.array([-0.1, 0.2, 5.1])

times = np.linspace(0, 0.002, 1200)
t0_true = 0.00085

v_mean_ion = 10000
v_sigma_ion = 2000

# USER-DEFINED IMPACT POINT
IMPACT_POINT = np.array([0.2, 3.3, 5.0])


# ============================================================
# IMPACT PULSE
# ============================================================

def impact_pulse(t, t0, tau_rise=8e-6, tau_decay=2.5e-4):
    y = np.zeros_like(t)
    mask = t >= t0
    tt = t[mask] - t0
    y[mask] = np.exp(-tt/tau_decay) - np.exp(-tt/tau_rise)
    return y


# ============================================================
# COSINE DISTRIBUTION
# ============================================================

def sample_cosine_direction(axis):

    axis = axis / np.linalg.norm(axis)

    while True:
        theta = np.random.rand() * np.pi / 2
        if np.random.rand() <= np.cos(theta):
            break

    phi = 2 * np.pi * np.random.rand()

    if abs(axis[2]) < 0.9:
        e1 = np.cross(axis, [0, 0, 1])
    else:
        e1 = np.cross(axis, [0, 1, 0])

    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    return (
        np.cos(theta)*axis +
        np.sin(theta)*(np.cos(phi)*e1 + np.sin(phi)*e2)
    )


# ============================================================
# LOAD + RECENTER STL
# ============================================================

def load_stl_geometry(path):

    m = stl_mesh.Mesh.from_file(path)

    verts = m.vectors.reshape(-1, 3)

    min_corner = verts.min(axis=0)
    max_corner = verts.max(axis=0)
    center = (min_corner + max_corner) / 2.0

    m.vectors -= center

    print("Original STL offset:", center)
    print("STL recentered to origin.\n")

    return m


# ============================================================
# FIND CLOSEST TRIANGLE
# ============================================================

def find_closest_triangle(mesh, point):

    min_dist = np.inf
    best_idx = None

    for i in range(len(mesh.vectors)):
        v0, v1, v2 = mesh.vectors[i]
        centroid = (v0 + v1 + v2) / 3

        dist = np.linalg.norm(point - centroid)

        if dist < min_dist:
            min_dist = dist
            best_idx = i

    return best_idx


# ============================================================
# BIOT–SAVART
# ============================================================

def biot_savart(q, v_vec, r_vec):

    R = np.linalg.norm(r_vec)

    if R < 1e-9:
        return np.zeros(3)

    return (mu0/(4*np.pi)) * q * np.cross(v_vec, r_vec) / (R**3)


# ============================================================
# FORWARD MODEL
# ============================================================

def simulate_plasma_plume(N, impact_point, normal):

    r_scm = SCM_OFFSET

    Bx = np.zeros_like(times)
    By = np.zeros_like(times)
    Bz = np.zeros_like(times)

    for _ in range(N):

        v_dir = sample_cosine_direction(normal)
        v_mag = np.random.normal(v_mean_ion, v_sigma_ion)
        v_vec = v_mag * v_dir

        r_vec = r_scm - impact_point

        B_pref = biot_savart(q_ion, v_vec, r_vec)

        t0_i = t0_true + np.random.randn()*1.5e-5
        pulse = impact_pulse(times, t0_i)

        Bx += B_pref[0] * pulse
        By += B_pref[1] * pulse
        Bz += B_pref[2] * pulse

    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    return Bx, By, Bz, Bmag


# ============================================================
# LOCAL SIMULATION
# ============================================================

def simulate_point(impact_point, normal):

    r_scm = SCM_OFFSET

    Bx = np.zeros_like(times)
    By = np.zeros_like(times)
    Bz = np.zeros_like(times)

    for _ in range(3000):

        v_dir = sample_cosine_direction(normal)
        v_mag = np.random.normal(v_mean_ion, v_sigma_ion)
        v_vec = v_mag * v_dir

        r_vec = r_scm - impact_point

        B_pref = biot_savart(q_ion, v_vec, r_vec)

        t0_i = t0_true + np.random.randn()*1.5e-5
        pulse = impact_pulse(times, t0_i)

        Bx += B_pref[0] * pulse
        By += B_pref[1] * pulse
        Bz += B_pref[2] * pulse

    return Bx, By, Bz


# ============================================================
# INVERSE LOCALIZATION
# ============================================================

def inverse_localization(mesh, Bx_meas, By_meas, Bz_meas):

    best_error = np.inf
    best_point = None
    best_triangle = None

    for i in range(len(mesh.vectors)):

        v0, v1, v2 = mesh.vectors[i]
        impact_point = (v0 + v1 + v2) / 3
        normal = mesh.normals[i]

        Bx_pred, By_pred, Bz_pred = simulate_point(impact_point, normal)

        error = (
            np.linalg.norm(Bx_pred - Bx_meas) +
            np.linalg.norm(By_pred - By_meas) +
            np.linalg.norm(Bz_pred - Bz_meas)
        )

        if error < best_error:
            best_error = error
            best_point = impact_point
            best_triangle = i

    return best_point, best_triangle, best_error


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Loading spacecraft STL...\n")

    mesh = load_stl_geometry(STL_PATH)

    print("SCM position:", SCM_OFFSET)

    tri_idx = find_closest_triangle(mesh, IMPACT_POINT)

    v0, v1, v2 = mesh.vectors[tri_idx]
    true_impact_point = (v0 + v1 + v2) / 3
    normal = mesh.normals[tri_idx]

    print("\n===== USER IMPACT =====")
    print("Requested:", IMPACT_POINT)
    print("Using triangle:", tri_idx)
    print("Actual impact:", true_impact_point)

    #  REAL DISTANCE CHECK
    R_true = np.linalg.norm(SCM_OFFSET - true_impact_point)
    print("True SCM distance:", R_true, "m")

    for N in ION_COUNTS:

        print("\n===================================")
        print("Running simulation with", N, "ions")
        print("===================================")

        Bx_meas, By_meas, Bz_meas, Bmag = simulate_plasma_plume(
            N, true_impact_point, normal
        )

        Bx_meas = uniform_filter1d(Bx_meas, 11)
        By_meas = uniform_filter1d(By_meas, 11)
        Bz_meas = uniform_filter1d(Bz_meas, 11)

        print("Running inverse localization...")

        impact_point, tri, error = inverse_localization(
            mesh, Bx_meas, By_meas, Bz_meas
        )

        R = np.linalg.norm(SCM_OFFSET - impact_point)

        print("\n===== INVERSE SOLUTION =====")
        print("Estimated triangle:", tri)
        print("Estimated coordinates:", impact_point)
        print("Distance SCM → impact:", R)
        print("Residual error:", error)

        print("\nComparison:")
        print("True triangle:", tri_idx)
        print("Triangle error:", abs(tri - tri_idx))

        plt.figure(figsize=(10,6))
        plt.plot(times, Bx_meas, label="Bx")
        plt.plot(times, By_meas, label="By")
        plt.plot(times, Bz_meas, label="Bz")
        plt.plot(times, Bmag, label="|B|", linewidth=2)
        plt.legend()
        plt.grid()
        plt.title(f"N = {N}")
        plt.show()
