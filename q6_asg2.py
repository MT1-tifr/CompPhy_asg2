import numpy as np
import matplotlib.pyplot as plt
g=10
def f1(t,x,y):
    return y
def f2(t,x,y):
    return -g

def euler_method(f1, f2, t0, x0, y0, h, t_max): #defining the algorithm for Euler method
    t_values=[t0] #creating an array of t values with t0 as the only entry 
    y_values=[y0]#creating an array of y values with y0 as the only entry
    x_values=[x0]
    while t_values[-1] < t_max:  #the index -1 gives the last element of the array
        
        t = t_values[-1]
        y = y_values[-1]
        x = x_values[-1]
        x_next = x + h * f1(t, x, y)
        y_next = y + h * f2(t, x, y)#calculating the value of y at next mesh point based on previous data
        t_values.append(t + h) #adding the next value in the array
        x_values.append(x_next)
        y_values.append(y_next)
    return t_values, x_values, y_values


def shooting_method(t0, t_max, x0, x1, y0, tol=1e-6, max_iter=100):
    family_of_curves = []
    for i in range(max_iter):
        t, x, _ = euler_method(f1, f2, t0, x0, y0, 0.5, t_max)
        family_of_curves.append(x)
        print(family_of_curves)
        res = x[-1] - x1
        if abs(res) < tol:
            print(f"Converged after {i} iterations.")
            return family_of_curves, t, x
        y0 -= res / 10  # adjust the initial guess using simple Newton's method
    print("Did not converge within maximum iterations.")
    return None, None, None
t0 = 0
t_max = 10
x1 = 0
y0=56
x0 = 0
family_of_curves, t, sol = shooting_method(t0, t_max, x0, x1, y0)

def exact_sol(t):
    return -5* t**2 + 50*t
t_values = np.linspace(0, 10, 100)
function_values = exact_sol(t_values)
    # Plot the solution and family of curves
plt.plot(t, sol, label='Numerical Solution')

n = len(family_of_curves)
for i in range(n-1):
    plt.plot(t, family_of_curves[i], linestyle='--', label=f'Guess {i+1}')
    
plt.plot(t_values, function_values,linestyle='--', label='Exact solution',color='red')

for i in range(5):
    y0 = i - 1
    t, x, _ = euler_method(f1, f2, t0, x0, y0, 0.5, t_max)
    plt.plot(t, x, label=f'ExtraGuess {i+1}' )
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Solution and Family of Curves')
plt.legend()
plt.grid(True)
plt.show()
    

    
