#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from pprint import pprint

def find_extremum(f, x_values, s, t):
    # print(x_values)
    # print(x_values[s], x_values[t])
    arg_max = argrelextrema(f(x_values[s:t]), np.greater)
    arg_max = np.array(arg_max[0])

    arg_min = argrelextrema(f(x_values[s:t]), np.less)
    arg_min = np.array(arg_min[0])

    if len(arg_min) == 0 and len(arg_max) == 0:
        res = s if abs(f(x_values[s])) > abs(f(x_values[t])) else t

    else:
        arg = s+ (arg_min if len(arg_min) else arg_max)
        arg = arg[0]
        if abs(f(x_values[s])) - abs(f(x_values[arg])) > 0:
            res = s
        elif abs(f(x_values[t])) - abs(f(x_values[arg])) > 0:
            res = t
        else:
            res =  arg

    return res


def find_all_extrema(f, N, poly, a, b, show_steps=False):
    x = np.linspace(a, b, 10000)
    error_function = lambda xi: poly(xi) - f(xi)

    roots = np.zeros(N + 1, dtype=int)
    j = 0
    for i in range(len(x)-1):
        if np.sign(error_function(x[i])) != np.sign(error_function(x[i+1])):
            if j >= len(roots):
                # print(":(((", i, x[i])
                break
            roots[j] = i
            j +=1
            # print(i, x[i])
    intervals = np.insert(roots, 0, 0)
    intervals = np.insert(intervals, len(intervals), len(x) -1)
    # print(roots)
    # print(x[roots])
    extrema = [None] * (N + 2)

    for i in range(len(intervals) -1):
        extr = find_extremum(error_function, x, intervals[i], intervals[i+1])
        # print(extr, x[extr])
        extrema[i] = extr

    extrema = np.array(extrema)

    if show_steps:
        plt.plot(x, error_function(x), label = 'error func')
        plt.scatter(x[intervals], error_function(x[intervals]), label="intervals")
        plt.scatter(x[extrema], error_function(x[extrema]), label="extrema")
        plt.legend()
        plt.show()

    # print(extrema, len(extrema))

    return x[extrema]

def exchange(f, N, poly, a, b, show_steps):
    new_x_values = find_all_extrema(f, N,poly, a, b, show_steps)
    errors = poly(new_x_values) - f(new_x_values)
    e_min, e_max = np.min(np.abs(errors)), np.max(np.abs(errors))
    return new_x_values, e_min, e_max

def solve_system(x_values, y_values):
    size = len(x_values)
    powers = np.arange(size-1)
    A = np.zeros((size, size))
    for i in range(size):
        A[i, :size-1] = x_values[i] ** powers

    a = np.ones(size)
    a[1::2] = -1
    A[:, size-1] = a

    solution = np.linalg.solve(A, y_values)

    return np.poly1d(solution[:size-1][::-1]), solution[size-1]

def remez(f, N, a, b, tolerance=1e-6, plot_steps=False):
    x_values = np.linspace(a, b, N+2)
    y_values = f(x_values)

    i=0

    if plot_steps:
        x = np.linspace(a, b, 1000)
        plt.plot(x, f(x), label='f(x)')
        plt.scatter(x_values, f(x_values), label = 'nodes')

        plt.legend()
        plt.show()

    while True:
        print("i: {}".format(i))
        poly, error = solve_system(x_values, y_values)



        x_values, e_min, e_max = exchange(f, N, poly, a, b, plot_steps)
        y_values = f(x_values)

        i+=1
        print("E_M / E_m = {}\nError = {}\n".format(e_max / e_min, error))
        if plot_steps:
            x = np.linspace(a, b, 1000)
            plt.plot(x, f(x), label='f(x)')
            plt.scatter(x_values, poly(x_values), label = 'nodes')
            plt.plot(x, poly(x), label='approx')

            plt.legend()
            plt.show()



        if (e_max / e_min) <= 1 + tolerance:
            break

        if i > 7:
            break


    x = np.linspace(a, b, 1000)
    plt.plot(x, f(x))
    plt.plot(x, poly(x))

    plt.show()
    print(poly)

#%%

f = lambda x: 1 / x
remez(f, 5, 1, 2, tolerance=0.05, plot_steps=True)
#%%

f = lambda x: np.exp(x) * np.sin(np.pi * x)
remez(f, 6, -0.5, 2, tolerance=0.05, plot_steps=True)
#%%

f = lambda x : np.sqrt(np.abs(x))
remez(f, 5, -1, 1, tolerance=0.05, plot_steps=True)

#%%

f = lambda x : (np.exp(x) * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * x))
remez(f, 5, 0.2, 0.8, tolerance=0.05, plot_steps=True)