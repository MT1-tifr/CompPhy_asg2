import numpy as np
import matplotlib.pyplot as plt
def f(t, u1, u2, u3):
    
    return (u1 + (2*u2) - (2*u3) + np.exp(-t)), (u2 +u3 - (2*np.exp(-t))), (u1 + (2*u2) + np.exp(-t)) #defining a function for the 2 differential equations

def runge_kutta4(f, t0, u10, u20, u30, h, tmax): #defining the algorithm for 4th order Runge-Kutta method with 2 first order differential equations

    t_values = [t0]
    u1_values = [u10]
    u2_values = [u20]
    u3_values = [u30]

    while t_values[-1] < tmax: #-1 index gives the last element of the array
        k1_u1, k1_u2, k1_u3 = f(t_values[-1], u1_values[-1], u2_values[-1], u3_values[-1]) #calculating all 4 k's for both y1 and y2
        k2_u1, k2_u2, k2_u3 = f(t_values[-1] + h/2, u1_values[-1] + h/2 * k1_u1, u2_values[-1] + h/2 * k1_u2, u3_values[-1] + h/2 * k1_u3)
        k3_u1, k3_u2, k3_u3 = f(t_values[-1] + h/2, u1_values[-1] + h/2 * k2_u1, u2_values[-1] + h/2 * k2_u2, u3_values[-1] + h/2 * k2_u3)
        k4_u1, k4_u2, k4_u3 = f(t_values[-1] + h, u1_values[-1] + h * k3_u1, u2_values[-1] + h * k3_u2, u3_values[-1] + h * k3_u3)

        u1_new = u1_values[-1] + h/6 * (k1_u1 + 2*k2_u1 + 2*k3_u1 + k4_u1) #computing the next y from the calculated k values and previous data
        u2_new = u2_values[-1] + h/6 * (k1_u2 + 2*k2_u2 + 2*k3_u2 + k4_u2)
        u3_new = u3_values[-1] + h/6 * (k1_u3 + 2*k2_u3 + 2*k3_u3 + k4_u3)

        t_values.append(t_values[-1] + h) #appending the new values in the array
        u1_values.append(u1_new)
        u2_values.append(u2_new)
        u3_values.append(u3_new)

    return t_values, u1_values, u2_values, u3_values

t0 = 0
u10 = 3
u20 = -1
u30 = 1
h = 0.1
tmax = 1

t_values, u1_values, u2_values, u3_values = runge_kutta4(f, t0, u10, u20, u30, h, tmax)

for i in range(len(t_values)):
    print(f"t = {t_values[i]:.2f}, u1'(x) = {u1_values[i]:.6f}, u2'(x) = {u2_values[i]:.6f}, u3'(x) = {u3_values[i]:.6f}")

plt.plot(t_values, u1_values, label='u1')
plt.plot(t_values, u2_values, label='u2')
plt.plot(t_values, u3_values, label='u3')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('Solution of the Initial Value Problem')
plt.legend()
plt.grid(True)
plt.show()
