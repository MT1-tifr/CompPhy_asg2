import numpy as np
import matplotlib.pyplot as plt

def f(t, x):
    return 1 / (x**2 + t**2)

def runge_kutta(f, t, x, h):
    k1 = h * f(t, x)
    k2 = h * f(t + 0.5 * h, x + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, x + 0.5 * k2)
    k4 = h * f(t + h, x + k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


x0 = 1
t0 = 0
t_final = 40
t_values = [t0]
x_values = [x0]
h = 0.5
while t_values[-1] < t_final:
    x_next = runge_kutta(f, t_values[-1], x_values[-1], h)
    t_values.append(t_values[-1] + h)
    x_values.append(x_next)

plt.plot(t_values, x_values)
plt.scatter(t_values, x_values)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Solution of dx/dt = 1/(x^2 + t^2)')
plt.grid(True)
plt.show()

# Printing the value of the solution at t = 3.5e6
print("Value of the solution at t = 3.5e6:", x_values[-1])
