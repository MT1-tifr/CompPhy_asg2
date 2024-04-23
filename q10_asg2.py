import numpy as np
import matplotlib.pyplot as plt


def f(t,y):
    return (y**2 + y)/t

def runge_kutta(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def adaptive_stepsize(f, t0, y0, tmax, h, delta):
    t_values = [t0]
    y_values = [y0]
    while ( t_values[-1] < tmax):
        y_next1 = runge_kutta(f, t_values[-1], y_values[-1], h)
        y_next1 = runge_kutta(f, t_values[-1]+h, y_values[-1]+h, h)
        y_next2 = runge_kutta(f, t_values[-1], y_values[-1], 2*h)
        error = np.abs(y_next2 - y_next1)/30
        
        if error != 0:
            rho = h * delta / error
            if rho > 1:
                h = 2*h
                t_values.append(t_values[-1] + 2*h)
                y_values.append(y_next2)
                continue
            else:
                h = h/2
                y_next2 = runge_kutta(f, t_values[-1], y_values[-1], h/2)
                t_values.append(t_values[-1] + h/2)
                y_values.append(y_next2)
                continue
        else:
            t_values.append(t_values[-1] + h)
            y_values.append(y_next1)
        
    return t_values, y_values

t0 = 1
tmax = 3
y0 = -2
delta = 1e-4
h = 0.5
t_values, y_values = adaptive_stepsize(f, t0, y0, tmax, h, delta)
print(t_values, y_values)
# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label='Solution')
plt.scatter(t_values, y_values, color='red', label='Mesh Points')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of y\' = (y^2 + y)/t with Adaptive Step-Size Control')
plt.legend()
plt.grid(True)
plt.show()
