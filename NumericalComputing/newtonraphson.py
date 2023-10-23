"""
    Implementation of the Newton-Raphson method to find a root of the function f: r^2 --> R^2 given by

        f(x_1,x_2) = (x_2^2 e^(x_1) - 3, 2x_2 e^(x_1) + 10 x_1^4,

    starting at a point x = (x_1, x_2).

    This program arose from exercise in 2.7.1 of the textbook S.J. Colley, Vector Calculus (4th Edition).
"""

import numpy as np
import math

""" 
    Class to calculate f, Df, and Df^(-1) (with f given above).
"""
class Func:
    def __init__(self):
        # Constructor.  f is the function R^2 --> R^2 of interest, with Df being its derivative and Df_inv
        # being Df^(-1).
        return

    def f(self,x):
        # The function of interest
        return np.array([
            x[1]**2 * math.exp(x[0]) - 3,
            2 * x[1] * math.exp(x[0]) + 10 * x[1]**4
        ])

    def Df(self,x):
        # Derivative of f at (x,y).
        return np.array([
            [ x[1]**2 * math.exp(x[0]), 2 * x[1] * math.exp(x[0])],
            [ 2 * x[1] * math.exp(x[0]), 2 * math.exp(x[0]) + 40 * x[1]**3]
        ])

    def Df_inv(self,x):
        # Inverse of derivative of f at (x,y).  This method throws an exception is Df is not invertible.
        # We'll calculate the inverse ourselves rather than rely on numpy.
        # Calculate the determinant of Df
        det_Df = 40 * x[1]**5 * math.exp(x[0]) - 2 * x[1]**2 * math.exp(2*x[0])

        if np.allclose(det_Df,np.array([[0,0],[0,0]])):
            # Df is singular and cannot be inverted.
            raise ValueError('Df is non-invertible')

        # At this point, we know Df is invertible.
        return det_Df**(-1) * np.array([
            [2 * math.exp(x[0]) + 40 * x[1]**3, - 2 * x[1] * math.exp(x[0])],
            [-2 * x[1] * math.exp(x[0]), x[1]**2 * math.exp(x[0])]
        ])

"""
    Class to run the Newton-Raphson method on a function R^2 --> R^2 with starting point at x = (x_1, x_2).
"""
class NewtonRaphson:

    def __init__(self,func,x):
        # Constructor.  func is the function of interest
        self._func = func

        # Points stores the points at each iteration
        self._points = [x]

        # iter_vals stores that value of func.f at each iteration
        self._iter_vals = [self._func.f(x)]

        # cur_iter stores the current iteration of the algorithm
        self._cur_iter = 0

    def run_iteration(self):
        # Run a single iteration of Newton-Raphson.
        self._cur_iter += 1
        x = self._points[-1]
        self._points.append(x - self._func.Df_inv(x) @ self._func.f(x))
        new_x = self._points[-1]
        self._iter_vals.append(self._func.f(new_x))

    def run_to_convergence(self, max_iters = 200, rtol = 1e-5, atol = 1e-8 ):
        # Run Newton-Raphson until convergence of f(x_n) to 0 or max_iters iterations.
        # Return True if there is convergence and false otherwise.
        # Convergence to zero uses np.allclose with rtol and atol as given.
        zero = np.array([0,0])
        converged = False
        for i in range(max_iters):
            self.run_iteration()
            if np.allclose(self._iter_vals[-1],zero,rtol,atol):
                converged = True
                break

        return converged

    # Class attributes
    @property
    def cur_iter(self):
        """ The last (i.e., current) iteration of Newton-Raphson. """
        return self._cur_iter

    @property
    def points(self):
        """ A list of points calculated using Newton-Raphson. This list is a copy of the one stored
            internally by the NewtonRaphson class, so you may modify this list as you wish. """
        return self._points.copy()

    @ property
    def iter_vals(self):
        """ A list of values of func based on the sequence of points determined by Newton-Raphson.  This
            list is a copy of the one stored internally by the NewtonRaphson class, so you may modify this
            list as you wish.
        """
        return self._iter_vals.copy()


if __name__ == '__main__':
    # Run Newton-Raphson on the function in the class Func at x = (1,-1).
    x = np.array([1,-1])

    f = Func()
    nr = NewtonRaphson(f,x)
    if nr.run_to_convergence(50):
        print(f'f(x_n) in Newton-Raphson coverged to 0 in {nr.cur_iter} iterations.')
    else:
        print('f(x_n) in Netwton-Raphson did not converge to 0.')

    # Print the results of each iteration.
    for i in range(nr.cur_iter + 1):
        print(f'x{i} = {nr.points[i]},\t\tf(x{i}) = {nr.iter_vals[i]}')
