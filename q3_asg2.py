import numpy as np
import matplotlib.pyplot as plt
def f(x, y1, y2):
    
    return y2, 2*y2 - y1 + x * (2.71828 ** x) - x #defining a function for the 2 differential equations

def runge_kutta4(f, x0, y10, y20, h, xmax): #defining the algorithm for 4th order Runge-Kutta method with 2 first order differential equations

    x_values = [x0]
    y1_values = [y10]
    y2_values = [y20]

    while x_values[-1] < xmax: #-1 index gives the last element of the array
        k1_y1, k1_y2 = f(x_values[-1], y1_values[-1], y2_values[-1]) #calculating all 4 k's for both y1 and y2
        k2_y1, k2_y2 = f(x_values[-1] + h/2, y1_values[-1] + h/2 * k1_y1, y2_values[-1] + h/2 * k1_y2)
        k3_y1, k3_y2 = f(x_values[-1] + h/2, y1_values[-1] + h/2 * k2_y1, y2_values[-1] + h/2 * k2_y2)
        k4_y1, k4_y2 = f(x_values[-1] + h, y1_values[-1] + h * k3_y1, y2_values[-1] + h * k3_y2)

        y1_new = y1_values[-1] + h/6 * (k1_y1 + 2*k2_y1 + 2*k3_y1 + k4_y1) #computing the next y from the calculated k values and previous data
        y2_new = y2_values[-1] + h/6 * (k1_y2 + 2*k2_y2 + 2*k3_y2 + k4_y2)

        x_values.append(x_values[-1] + h) #appending the new values in the array
        y1_values.append(y1_new)
        y2_values.append(y2_new)

    return x_values, y1_values, y2_values

# Specifying the initial conditions
x0 = 0
y10 = 0
y20 = 0
h = 0.1
xmax = 1

# Solving the ODE using the 4th-order Runge-Kutta method
x_values, y1_values, y2_values = runge_kutta4(f, x0, y10, y20, h, xmax)

# Printing the results
for i in range(len(x_values)):
    print(f"x = {x_values[i]:.2f}, y(x) = {y1_values[i]:.6f}, y'(x) = {y2_values[i]:.6f}")

#plotting the values obtained
plt.figure(figsize=(10, 6))
plt.plot(x_values,y1_values)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution using Runge-Kutta Method')
plt.legend()
plt.grid(True)
plt.show()
