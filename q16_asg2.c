#include <stdio.h>
#include <math.h>

// Exact solution function
double exact_solution(double t) {
    return pow(t + 1, 2) - 0.5 * exp(t);
}

int main() {
    double h = 0.2;
    double t_max = 2.0; 
    double y0 = 0.5; 
    int steps = (int)(t_max / h); 

    double t = 0.0;
    double y = y0; 
    double error;

    printf("Step\tt\t\tExact Solution\t\tApproximation\t\tError\t\tError Bound\n");
    for (int i = 0; i <= steps; i++) {
        double exact = exact_solution(t); 
        error = fabs(exact - y); 

           
        printf("%d\t%.2f\t\t%.6f\t\t%.6f\t\t%.6f\t\n", i, t, exact, y, error);

        y = y + h * (y - t * t + 1);
        t = t + h;
    }

    return 0;
}
