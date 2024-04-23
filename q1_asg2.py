import numpy as np
import matplotlib.pyplot as plt


def euler_method1(x0, y0, h, x_max): 
    x_values=[x0]
    y_values=[y0]
    while x_values[-1] < x_max:
        x = x_values[-1]
        y = y_values[-1]
        y_next = y / (1 + 9*h)
        x_values.append(x + h)
        y_values.append(y_next)
    return x_values, y_values

x1,y1 = euler_method1(0, np.exp(1), 0.01, 1)

def f2(x,y):
    return -20*(y-x)**2 + 2*x

def df2(x,y):
    return -40*(y-x)
def euler_method2(x0, y0, h, x_max):
    x_values = [x0]
    y_values = [y0]
    N = int((x_max-x0)*1.0/h)

    for i in range(N):
        t = 1
        y0 = 0
        while(t>0.001):
            y1 = y0 - (h*f2(x0+i*h,y0)+y_values[-1]-y0)/(h*df2(x0+i*h,y0)-1)   # SOLVING FOR THE ROOT BY NEWTON'S METHOD 
            t = np.abs(y1-y0)
            y0 = y1
    
        y_values.append(y0)
        x_values.append(x0+(i+1)*h)
    return x_values, y_values

x2,y2 = euler_method2(0, 1/3, 0.01, 1)

    
plt.figure(figsize=(10, 6))
plt.plot(x1,y1, label='Numerical Solution of part A')
plt.plot(x2,y2, label='Numerical Solution of part B ')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solutions using Euler\'s Method')
plt.legend()
plt.grid(True)
plt.show()
