# Library
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from stl.mesh import Mesh
from scipy.integrate import solve_ivp

# Parameters
inital_state = [
    np.pi/1.5, #theta_1
    np.pi/1.5, #theta_2
    0,       #omega_1
    0        #omega_2
]
density = 1.0
alpha = 0.0
framerate = 30
duration = 100.0
g = 9.8

# モデルの情報を獲得（パスは任意）
upper_path = "upper.stl"
lower_path = "lower.stl"
upper_model = Mesh.from_file(upper_path)
lower_model = Mesh.from_file(lower_path)
upper_vol, upper_cog, upper_imat = upper_model.get_mass_properties()
lower_vol, lower_cog, lower_imat = lower_model.get_mass_properties()
r_1, r_2 = float(np.linalg.norm(upper_cog)), float(np.linalg.norm(lower_cog))
m_1, m_2 = float(upper_vol * density), float(lower_vol * density)
I_1, I_2 = float(upper_imat[0, 0]), float(lower_imat[0, 0])
U = np.array([-0.086, 0.024, -0.367])  # upper.stl 内の接続点
L = np.array([ 0.067, 0.337,  0.886])  # lower.stl 内の接続点
l = float(np.linalg.norm(U - 0))
print(f"{r_1=}\n{r_2=}\n{m_1=}\n{m_2=}\n{I_1=}\n{I_2=}")

t_span = (0.0, duration)
t_eval = np.linspace(0.0, duration, int(duration * framerate))

A = m_1 * r_1**2 + m_2 * l**2 + I_1
D = m_2 * r_2**2 + I_2

def B(t1, t2):
    return m_2 * l * r_2 * np.cos(t1 + alpha - t2)

def E(t1, t2, o2):
    return (
        -m_1 * g * r_1 * np.sin(t1)
        -m_2 * g * l   * np.sin(t1 + alpha)
        -m_2 * l * r_2 * (o2**2) * np.sin(t1 + alpha - t2)
    )

def F(t1, t2, o1):
    return (
        -m_2 * g * r_2 * np.sin(t2)
        +m_2 * l * r_2 * (o1**2) * np.sin(t1 + alpha - t2)
    )

def eom(t, X):
    t1, t2, o1, o2 = X
    b = B(t1, t2)
    c = b
    e = E(t1, t2, o2)
    f = F(t1, t2, o1)
    den = A * D - b * c
    a1 = (e * D - b * f) / den
    a2 = (A * f - e * c) / den
    return [o1, o2, a1, a2]

sol = solve_ivp(
    eom, t_span, inital_state, t_eval=t_eval,
    rtol=1e-9, atol=1e-9
)

theta1 = sol.y[0]
theta2 = sol.y[1]
omega1 = sol.y[2]
omega2 = sol.y[3]

plt.figure()
plt.plot(sol.t, theta1, label="θ1")
plt.plot(sol.t, theta2, label="θ2")
plt.legend()
plt.xlabel("t[s]")
plt.ylabel("θ[rad]")
plt.show()

plotter = pv.Plotter(notebook=True, window_size=[512, 512])
upper: pv.DataSet = pv.read(upper_path)
lower: pv.DataSet = pv.read(lower_path)
lower.translate(U - L, inplace=True)
plotter.add_mesh(upper)
plotter.add_mesh(lower)
plotter.camera.position = (10, 10, 0)
plotter.camera.focal_point = (0, 0, 0)
plotter.open_movie("result.mp4", framerate=framerate)
t_1_diffs = np.diff(sol.y[0], prepend=[0])
t_2_diffs = np.diff(sol.y[1], prepend=[0])
for (t1, dt1, dt2) in zip(sol.y[0], t_1_diffs, t_2_diffs):
    upper.rotate_x(np.rad2deg(dt1), inplace=True)
    lower.rotate_x(np.rad2deg(dt1), inplace=True)
    ct, st = np.cos(t1), np.sin(t1)
    J = np.array([
        U[0],
        U[1]*ct - U[2]*st,
        U[1]*st + U[2]*ct
    ])
    lower.rotate_x(np.rad2deg(dt2 - dt1), point=tuple(J), inplace=True)
    plotter.write_frame()
plotter.close()