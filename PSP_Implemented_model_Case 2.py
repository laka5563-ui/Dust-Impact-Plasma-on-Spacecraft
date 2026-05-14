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
# TIME
# ============================================================

times = np.linspace(0.85, 0.90, 1200)
dt = times[1] - times[0]
t0_true = 0.862

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

def generate_measured_data(mesh, SCM):

    pulse = impact_pulse(times, t0_true)

    # -----------------------------
    # GAUSSIAN VELOCITY (UNCHANGED)
    # -----------------------------
    centroid = np.mean(mesh.vectors.reshape(-1,3), axis=0)
    distance = np.linalg.norm(SCM - centroid)

    mean_v = 6000
    std_v  = 0.15 * mean_v

    velocities = np.random.normal(mean_v, std_v, 2000)
    velocities = velocities[velocities > 0]

    tof = distance / velocities
    spread_time = np.std(tof)
    sigma = spread_time / dt

    pulse = gaussian_filter1d(pulse, sigma)

    # -----------------------------
    # MAGNETIC SIGNAL
    # -----------------------------
    Bx = -0.25e-9 * pulse
    By = -0.20e-9 * pulse
    Bz =  0.05e-9 * pulse

    Bx += -0.18e-9 * np.exp(-((times - t0_true)/0.0005)**2)
    By += -0.12e-9 * np.exp(-((times - (t0_true+0.00015))/0.0007)**2)
    Bz +=  0.04e-9 * np.exp(-((times - (t0_true-0.0001))/0.0006)**2)

    decay = np.exp(-(times - t0_true)/0.008)
    decay[times < t0_true] = 0

    Bx += -0.06e-9 * decay
    By += -0.025e-9 * decay
    Bz +=  0.015e-9 * decay

    osc = 0.015e-9 * np.sin(600*(times - t0_true)) * np.exp(-(times-t0_true)/0.01)
    osc[times < t0_true] = 0

    Bx += osc
    By += 0.8 * osc
    Bz += 0.3 * osc

    noise = 0.01e-9
    Bx += noise*np.random.randn(len(times))
    By += noise*np.random.randn(len(times))
    Bz += noise*np.random.randn(len(times))

    return Bx, By, Bz

# ============================================================
# BIOT-SAVART
# ============================================================

def biot_savart_segment(r_obs, r1, r2):

    dl = r2 - r1
    B = np.zeros(3)

    for s in np.linspace(0,1,10):
        point = r1 + s*dl
        r = r_obs - point
        r_norm = np.linalg.norm(r)

        if r_norm < 1e-9:
            continue

        dB = (mu0/(4*np.pi)) * np.cross(dl/10, r) / (r_norm**3)
        B += dB

    return B

# ============================================================
# GEOMETRY
# ============================================================

def load_stl_geometry(path):
    m = stl_mesh.Mesh.from_file(path)
    verts = m.vectors.reshape(-1,3)
    center = (verts.min(axis=0) + verts.max(axis=0))/2
    m.vectors -= center
    return m

# ============================================================
# INVERSE SOLVER
# ============================================================

def estimate_impact_inverse(mesh, SCM, B_peak):

    best_error = np.inf
    best_point = None
    best_tri = -1

    all_points = []
    all_errors = []

    for i, tri in enumerate(mesh.vectors):

        p = np.mean(tri, axis=0)

        B_model = biot_savart_segment(SCM, tri[0], tri[1])

        scale = np.dot(B_peak, B_model)/(np.dot(B_model,B_model)+1e-20)
        B_scaled = scale * B_model

        error = np.linalg.norm(B_peak - B_scaled)

        all_points.append(p)
        all_errors.append(error)

        if error < best_error:
            best_error = error
            best_point = p
            best_tri = i

    return best_point, best_tri, best_error, np.array(all_points), np.array(all_errors)

# ============================================================
# VALIDATION
# ============================================================

def forward_model_error(point, SCM, B_peak, mesh):

    min_dist = np.inf
    tri_idx = 0

    for i, tri in enumerate(mesh.vectors):
        p = np.mean(tri, axis=0)
        d = np.linalg.norm(point - p)

        if d < min_dist:
            min_dist = d
            tri_idx = i

    tri = mesh.vectors[tri_idx]

    B_model = biot_savart_segment(SCM, tri[0], tri[1])

    scale = np.dot(B_peak, B_model)/(np.dot(B_model,B_model)+1e-20)
    B_scaled = scale * B_model

    return np.linalg.norm(B_peak - B_scaled)

def get_top_solutions(points, errors, n=5):
    idx = np.argsort(errors)[:n]
    return points[idx], errors[idx], idx

def sensitivity_test(mesh, SCM, B_peak):
    noise = 0.05 * B_peak * np.random.randn(3)
    B_new = B_peak + noise
    pt, _, _, _, _ = estimate_impact_inverse(mesh, SCM, B_new)
    return pt

# ============================================================
# ION ESTIMATION
# ============================================================

def estimate_ions(Bmag, distance):

    peak_idx = np.argmax(Bmag)

    window = 15
    start = max(0, peak_idx-window)
    end   = min(len(Bmag), peak_idx+window)

    B_local = Bmag[start:end]

    baseline = np.mean(Bmag[:50])
    B_local = B_local - baseline
    B_local[B_local < 0] = 0

    I = (2*np.pi*distance*B_local)/mu0
    I[I < 0.1*np.max(I)] = 0

    Q = np.sum(I)*dt

    efficiency = 0.05
    ions = (Q*efficiency)/q_ion

    return ions

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    STL_PATH = r"C:/Users/klpra/OneDrive/Desktop/CU Boulder/Academics/Spring 2026/IS/PSP_CAD/PSP_CAD/PSP_Simplified.stl"

    mesh = load_stl_geometry(STL_PATH)
    SCM = np.array([-0.3, -0.5, 4.3])

    Bx, By, Bz = generate_measured_data(mesh, SCM)

    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    peak_idx = np.argmax(Bmag)
    peak_time = times[peak_idx]

    B_peak = np.array([Bx[peak_idx], By[peak_idx], Bz[peak_idx]])

    impact_point, tri_idx, error, all_points, all_errors = estimate_impact_inverse(mesh, SCM, B_peak)

    distance = np.linalg.norm(SCM - impact_point)
    ions = estimate_ions(Bmag, distance)

    forward_err = forward_model_error(impact_point, SCM, B_peak, mesh)
    top_pts, top_errs, top_idx = get_top_solutions(all_points, all_errors)
    new_pt = sensitivity_test(mesh, SCM, B_peak)

    print("\n===== FORWARD MODEL CHECK =====")
    print("Difference:", forward_err)

    print("\n===== TOP 5 SOLUTIONS =====")
    for i in range(len(top_pts)):
        print(f"Error: {top_errs[i]:.3e}, Triangle: {top_idx[i]}, Point: {top_pts[i]}")

    print("\n===== SENSITIVITY TEST =====")
    print("Original impact:", impact_point)
    print("New impact:", new_pt)

    print("\n===== IMPACT ESTIMATION =====")
    print("Triangle:", tri_idx)
    print("Coordinates:", impact_point)
    print("Distance:", distance)
    print("Residual:", error)
    print("Ions:", ions)

# ============================================================
# TOP-5 SOLUTION CLUSTER
# ============================================================

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# All candidate points (light)
ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2],
           c='lightgray', s=5, alpha=0.3)

# Top 5
ax.scatter(top_pts[:,0], top_pts[:,1], top_pts[:,2],
           c='red', s=80, label='Top 5 Solutions')

# Best solution
ax.scatter(impact_point[0], impact_point[1], impact_point[2],
           c='blue', s=120, label='Best Solution')

# SCM location 
ax.scatter(SCM[0], SCM[1], SCM[2],
           c='green', s=120, label='SCM')

ax.set_title("Top-5 Impact Solution Cluster")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()

plt.tight_layout()
plt.show()


# ============================================================
# ERROR DISTRIBUTION MAP
# ============================================================

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2],
                     c=all_errors, cmap='viridis', s=10)

# Highlight best point
ax.scatter(impact_point[0], impact_point[1], impact_point[2],
           c='red', s=120, label='Minimum Error')

cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
cbar.set_label('Residual Error')

ax.set_title("Residual Error Distribution Across Surface")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12,4))

plt.plot(times, Bx*1e9, label="Bx")
plt.plot(times, By*1e9, label="By")
plt.plot(times, Bz*1e9, label="Bz")
plt.plot(times, Bmag*1e9, label="|B|", linewidth=2)

plt.axvline(peak_time, linestyle='--')
plt.legend()
plt.grid()
plt.xlabel("Time (s)")
plt.ylabel("SCM (nT)")

plt.show()
