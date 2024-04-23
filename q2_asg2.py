import numpy as np
import matplotlib.pyplot as plt

def f(t, y):
    return y/t - (y/t)**2

def euler_method(f, t0, y0, h, t_max): #defining the algorithm for Euler method
    t_values=[t0] #creating an array of t values with t0 as the only entry 
    y_values=[y0]#creating an array of y values with y0 as the only entry 
    while t_values[-1] < t_max:  #the index -1 gives the last element of the array
        
        t = t_values[-1]
        y = y_values[-1]
        y_next = y + h * f(t, y)#calculating the value of y at next mesh point based on previous data
        t_values.append(t + h) #adding the next value in the array
        y_values.append(y_next)

    return t_values, y_values

def analytical_solution(t):
    return t / (1 + np.log(t)) # the given analytical solution

t0 = 1        # given parameters
t_max = 2
y0 = 1
h = 0.1

t_euler, y_euler = euler_method(f, t0, y0, h, t_max) #computing Euler solution

y_true = [analytical_solution(t) for t in t_euler] # Computing analytical solution

abs_err = [abs(y_true[i] - y_euler[i]) for i in range(len(t_euler))] #computing absolute error
rel_err = [abs_err[i] / y_true[i] for i in range(len(t_euler))] #computing relative error
# Printing the absolute and relative errors
for i in range(len(t_euler)):
    print("At t = ",t_euler[i],":")
    print("  Absolute Error:", abs_err[i])
    print("  Relative Error:", rel_err[i])
    
# Plotting the numerical and analytical solutions
plt.figure(figsize=(10, 6))
plt.plot(t_euler, y_euler, label='Numerical Solution using Euler\'s Method)')
plt.plot(t_euler, y_true, label='Analytical Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Numerical and Analytical Solutions')
plt.legend()
plt.grid(True)
plt.show()


