import numpy as np
import matplotlib.pyplot as plt

def relaxation_method(t1, g, n=100, tol=1e-3):
    h = t1 / (n - 1)  
    x = np.zeros(n)   
    x[-1] = 0         # Applying boundary condition x = 0 at t = t1
    
    while True:
        x_new = np.zeros(n)
        x_new[0] = 0  # Applying boundary condition x = 0 at t = 0
        
        for i in range(1, n - 1):
            x_new[i] = 0.5 * (x[i - 1] + x[i + 1] + h**2 * g)
        
        if np.max(np.abs(x_new - x)) < tol:
            break
        
        x = x_new  
    
    return x

t1 = 10
g = 10

x_numerical = relaxation_method(t1, g)
t = np.linspace(0, t1, len(x_numerical))
)
def exact_solution(t):
    return (g / 2) * t * (t1 - t)

# Generating candidate solutions (assuming a linear function with different slopes)
candidate_slopes = np.linspace(1.5, 2.5, 5)
x_candidates = [exact_solution(t) * slope for slope in candidate_slopes]

plt.figure(figsize=(10, 8))
plt.plot(t, exact_solution(t), label='Exact Solution', linestyle='--')
plt.plot(t, x_numerical, label='Numerical Solution', linestyle='-')
for i, candidate in enumerate(x_candidates):
    plt.plot(t, candidate, label=f'Candidate {i+1}')

plt.xlabel('t')
plt.ylabel('x')
plt.title('Comparison of Exact, Numerical, and Candidate Solutions')
plt.grid(True)
plt.legend()
plt.show()
