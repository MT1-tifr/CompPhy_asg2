import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f1(t, y):
    return t * np.exp(3 * t) - 2 * y
def y_analytical1(t):
    return (1/25) * np.exp(-2*t) * (1 + np.exp(5*t) * (-1 + 5*t))

def f2(t, y):
    return 1 - (t - y)**2
def y_analytical2(t):
    return (1 - 3* t + t**2) / (-3 + t)

def f3(t, y):
    return 1 + y / t
def y_analytical3(t):
    return t * (2 + np.log(t))

def f4(t, y):
    return np.cos(2 * t) + np.sin(3 * t)
def y_analytical4(t):
    return (1/6) * (8 - 2 * np.cos(3 * t) + 3 * np.sin(2 * t))

t_span1 = [0, 1]
y0_1 = [0]
t_span2 = [2, 3]
y0_2 = [1]
t_span3 = [1, 2]
y0_3 = [2]
t_span4 = [0, 1]
y0_4 = [1]

sol1 = solve_ivp(f1, t_span1, y0_1, t_eval=np.linspace(t_span1[0], t_span1[1], 100))
sol2 = solve_ivp(f2, t_span2, y0_2, t_eval=np.linspace(t_span2[0], t_span2[1], 100))
sol3 = solve_ivp(f3, t_span3, y0_3, t_eval=np.linspace(t_span3[0], t_span3[1], 100))
sol4 = solve_ivp(f4, t_span4, y0_4, t_eval=np.linspace(t_span4[0], t_span4[1], 100))


plt.figure(figsize=(12, 10))

# Plot for problem 1
plt.subplot(2, 2, 1)
plt.plot(sol1.t, sol1.y[0], label='Numerical Solution')
plt.plot(sol1.t, y_analytical1(sol1.t), label='Analytical Solution', linestyle='--')
plt.xlabel('t')
plt.ylabel('y')
plt.title(' y\' = te^(3t) - 2y, y(0) = 0')
plt.legend()
plt.grid(True)

# Plot for problem 2
plt.subplot(2, 2, 2)
plt.plot(sol2.t, sol2.y[0], label='Numerical Solution')
plt.plot(sol2.t, y_analytical2(sol2.t), label='Analytical Solution', linestyle='--')
plt.xlabel('t')
plt.ylabel('y')
plt.title(' y\' = 1 - (t - y)^2, y(2) = 1')
plt.legend()
plt.grid(True)

# Plot for problem 3
plt.subplot(2, 2, 3)
plt.plot(sol3.t, sol3.y[0], label='Numerical Solution')
plt.plot(sol3.t, y_analytical3(sol3.t), label='Analytical Solution', linestyle='--')
plt.xlabel('t')
plt.ylabel('y')
plt.title(' y\' = 1 + y/t, y(1) = 2')
plt.legend()
plt.grid(True)

# Plot for problem 4
plt.subplot(2, 2, 4)
plt.plot(sol4.t, sol4.y[0], label='Numerical Solution')
plt.plot(sol4.t, y_analytical4(sol4.t), label='Analytical Solution', linestyle='--')
plt.xlabel('t')
plt.ylabel('y')
plt.title(' y\' = cos(2t) + sin(3t), y(0) = 1')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
