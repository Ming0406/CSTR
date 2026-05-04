import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cstr_plant import CSTRSystem
from rbf_nn import RBFNetwork
from controller import CSTRController

plant = CSTRSystem()
ctrl = CSTRController(delta=plant.delta)

def total_dynamics(t, y_global):
    # y_global : [x1, x2, W1_0...W1_6, W2_0...W2_6]
    x1, x2 = y_global[0], y_global[1]
    W1 = y_global[2:9]
    W2 = y_global[9:16]
    yd, dot_yd = plant.get_reference_signal(t)
    kb = 0.25 + 0.05 * np.sin(t)     # Define kb(t)
    dot_kb = 0.05 * np.cos(t)
    u, dW1_dt, dW2_dt, alpha1 = ctrl.compute(x1, x2, yd, dot_yd, W1, W2, kb, dot_kb)
    dx_dt = plant.dynamics(t, [x1, x2], u)
    return np.concatenate([dx_dt, dW1_dt, dW2_dt])

# Initialization
x0 = [0.18, 0.5]  # conc. and temp.
W1_0 = np.zeros(7) # Set initial weight as 0
W2_0 = np.zeros(7)
y0_global = np.concatenate([x0, W1_0, W2_0])

# SIM
t_span = (0, 30)
t_eval = np.linspace(0, 30, 1000)
sol = solve_ivp(total_dynamics, t_span, y0_global, t_eval=t_eval, method='RK45')

# Result visualisation
t = sol.t
x1, x2 = sol.y[0], sol.y[1]
yd_plot = 0.1 * np.sin(t) + 0.2
kb_plot = 0.25 + 0.05 * np.sin(t)

plt.figure(figsize=(10, 8))

# pic.1: conc. and time-varying constraint
plt.subplot(2, 1, 1)
plt.plot(t, x1, 'r', label='Actual $x_1$ (Conc.)', linewidth=2)
plt.plot(t, yd_plot, 'b--', label='Desired $y_d$')
plt.plot(t, yd_plot + kb_plot, 'g:', label='Upper Bound')
plt.plot(t, yd_plot - kb_plot, 'g:', label='Lower Bound')
plt.title('CSTR Tracking Performance with Time-Varying Constraints')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)

# pic.2: u
u_history = []
for i in range(len(t)):
    yd, d_yd = plant.get_reference_signal(t[i])
    kb_i = 0.25 + 0.05 * np.sin(t[i])
    d_kb_i = 0.05 * np.cos(t[i])
    u_i, _, _, _ = ctrl.compute(sol.y[0,i], sol.y[1,i], yd, d_yd, sol.y[2:9,i], sol.y[9:16,i], kb_i, d_kb_i)
    u_history.append(u_i)

plt.subplot(2, 1, 2)
plt.plot(t, u_history, 'k', label='Control Input $u$')
plt.ylabel('Coolant Flow')
plt.xlabel('Time (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
print("SIMULATION COMPLETE")