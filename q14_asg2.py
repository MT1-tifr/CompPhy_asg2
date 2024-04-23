import numpy as np
import matplotlib.pyplot as plt

def f(t, y1, y2):
    return y2, ((2 * y2/t) - (2*y1/(t**2)) + (t * np.log(t)))

def euler_method(f, t0, y10, y20, h, t_max): #defining the algorithm for Euler method
    t_values=[t0] #creating an array of t values with t0 as the only entry 
    y1_values=[y10]#creating an array of y1 values with y10 as the only entry
    y2_values= [y20]#creating an array of y2 values with y20 as the only entry
    
    while t_values[-1] < t_max:  #the index -1 gives the last element of the array
        
        t = t_values[-1]
        y1 = y1_values[-1]
        y2 = y2_values[-1]
        y11, y22 =  f(t, y1, y2)
        y1_next, y2_next = y1 + h*y11, y2 + h*y22 #calculating the value of y at next mesh point based on previous data
        t_values.append(t + h) #adding the next value in the array
        y1_values.append(y1_next)
        y2_values.append(y2_next)

    return t_values, y1_values, y2_values

def analytical_solution(t):
    return (7*t/4) + (t**3)*np.log(t)/2 - (3/4)* t**3

t0 = 1      
t_max = 2
y10 = 1
y20 = 0
h = 0.001

t_euler, y1_euler, y2_euler = euler_method(f, t0, y10, y20, h, t_max) #computing Euler solution

y_true = [analytical_solution(t) for t in t_euler] # Computing analytical solution

plt.figure(figsize=(10, 6))
plt.plot(t_euler, y1_euler, label='Numerical Solution using Euler\'s Method)')
plt.plot(t_euler, y_true, label='Analytical Solution', linestyle='--')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Numerical and Analytical Solutions')
plt.legend()
plt.grid(True)
plt.show()
