import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def f1(x, y):
    return np.vstack((y[1], -np.exp(-2*y[0])))
def bc1(ya, yb):
    return np.array([ya[0], yb[0] - np.log(2)])

def f2(x, y):
    return np.vstack((y[1], y[1]*np.cos(x) - y[0]*np.log(y[0])))
def bc2(ya, yb):
    return np.array([ya[0] - 1, yb[0] - np.exp(1)])

def f3(x, y):
    return np.vstack((y[1], -(2*y[1]**3 + y[0]**2*y[1])/np.cos(x)))
def bc3(ya, yb):
    return np.array([ya[0] - 2**(-1/4), yb[0] - (12**0.25)/2])

def f4(x, y):
    return np.vstack((y[1], 0.5 - 0.5*y[1]**2 - 0.5*y[0]*np.sin(x)))
def bc4(ya, yb):
    return np.array([ya[0] - 2, yb[0] - 2])


x1 = np.linspace(1, 2, 100)
y_guess1 = np.zeros((2, x1.size))
sol1 = solve_bvp(f1, bc1, x1, y_guess1)

x2 = np.linspace(0, 0.5*np.pi, 100)
y_guess2 = np.zeros((2, x2.size))
y_b = np.zeros((2, x2.size))
y_b[0] = np.exp(1)  #done as given in scipy manual, it says we're computing the other solution using different initial condition
sol21 = solve_bvp(f2, bc2, x2, y_guess2)
sol22 = solve_bvp(f2, bc2, x2, y_b)

x3 = np.linspace(np.pi/4, np.pi/3, 100)
y_guess3 = np.zeros((2, x3.size))
sol3 = solve_bvp(f3, bc3, x3, y_guess3)

x4 = np.linspace(0, np.pi, 100)
y_guess4 = np.zeros((2, x4.size))
sol4 = solve_bvp(f4, bc4, x4, y_guess4)


plt.figure(figsize=(12, 10))

plt.subplot(221)
plt.plot(sol1.x, sol1.y[0])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Problem 1: y\'\' = -e^(-2y)')
plt.grid()

plt.subplot(222)
plt.plot(sol21.x, sol21.y[0])
plt.plot(sol22.x, sol22.y[0])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Problem 2: y\'\' = y\'*cos(x) - y*log(y)')
plt.grid()

plt.subplot(223)
plt.plot(sol3.x, sol3.y[0])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Problem 3: y\'\' = -(2(y\')^3 + y^2*y\')*sec(x)')
plt.grid()

plt.subplot(224)
plt.plot(sol4.x, sol4.y[0])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Problem 4: y\'\' = 0.5 - 0.5*(y\')^2 - 0.5*y*sin(x)')
plt.grid()

plt.tight_layout()
plt.show()
