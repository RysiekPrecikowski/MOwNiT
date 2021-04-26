import numpy as np
from sympy import *
from sympy.abc import x
import matplotlib.pyplot as plt
from enum import IntEnum


def my_integrate(f, m, mode, a=-1, b=1):
    def fk(k):
        return lf(a + k * h)

    h = (b - a) / (m)
    lf = lambdify(x, f)

    if mode == Mode.MIDPOINT:
        res = 0
        for k in range(m):
            res += fk(k + 0.5)
        res *= h

    elif mode == Mode.TRAPEZOIDAL:
        res = fk(0) + fk(m)
        for k in range(1, m):
            res += 2 * fk(k)
        res *= h / 2

    elif mode == Mode.SIMPSON_PARABOLIC:
        res = fk(0) + fk(m)
        for k in range(1, m):
            if k % 2 == 0:
                res += 2 * fk(k)
            else:
                res += 4 * fk(k)
        res *= h / 3

    elif mode == Mode.SIMPSON_CUBIC:
        res = fk(0) + fk(m)

        for k in range(1, m):
            if k % 3 == 0:
                res += 2 * fk(k)
            else:
                res += 3 * fk(k)
        res *= 3 * h / 8

    else:
        res = Gauss_Legendre(f, m, a, b)

    return res


def Gauss_Legendre(f, m, a=-1, b=1):
    p = legendre_poly(m, x, polys=True)
    r = p.real_roots()
    p_diff = diff(p, x)

    lf = lambdify(x, f)
    res = 0

    interval_sum = (a + b) / 2
    interval_sub = (b - a) / 2

    for xd in r:
        if xd == 0 or type(xd) == Mul:
            x_v = xd
        else:
            x_v = xd.eval_rational()
        w = 2 / ((1 - x_v ** 2) * (p_diff(x_v) ** 2))
        res += w * lf(interval_sub * x_v + interval_sum)

    res *= interval_sub

    return float(res)


class Mode(IntEnum):
    MIDPOINT = 0
    TRAPEZOIDAL = 1
    SIMPSON_PARABOLIC = 2
    SIMPSON_CUBIC = 3
    GAUSS_LEGENDRE = 4



def error(f, n, mode, a=-1, b=1):
    true_val = integrate(f, (x, a, b))
    integration_result = my_integrate(f, n, mode, a, b)
    return abs(true_val - integration_result) / true_val


def plot_error(f, mode, s, t, show=True, a=-1, b=1):
    points = []
    # print(mode)
    for i in range(s, t):
        # print(i, "------>", float(error(f, i, mode)))
        points.append((i + 1, error(f, i, mode, a, b)))

    plt.plot([x[0] for x in points], [x[1] for x in points], label=mode)
    plt.scatter([x[0] for x in points], [x[1] for x in points], s=12)
    if show:
        plt.title("relative errors for " + str(mode))
        plt.xlabel("evaluation points")
        plt.ylabel("relative error")
        plt.grid()
        plt.legend()
        plt.show()


def plot_errors(f, s, t, show_one_by_one=False, a=-1, b=1):
    for mode in Mode:
        plot_error(f, mode, s, t, show_one_by_one, a, b)

    if not show_one_by_one:
        plt.title("relative errors for all modes")
        plt.xlabel("evaluation points")
        plt.ylabel("relative error")
        plt.grid()
        plt.legend()
        plt.show()



def print_integrations(f, s, t, a=-1, b=1):
    true_val = integrate(f, (x, a, b))
    print("true val", true_val, "=", float(true_val))
    print()

    for mode in Mode:
        print()
        print("*********   MODE ", mode, "   *********")
        print()
        for n in range(s, t):
            print("n", n)
            # print()
            res = my_integrate(f, n, mode, a, b)
            print("int =", res)
            print("error = ", float(error(f, n, mode)))
            print()


def plot_integration(f, mode, s, t, show=True, a=-1, b=1):
    points = []
    for i in range(s, t):

        points.append((i + 1, my_integrate(f, i, mode, a, b)))
    plt.plot([p[0] for p in points], [p[1] for p in points], label=mode)
    plt.scatter([p[0] for p in points], [p[1] for p in points], s=12)


    if show:
        true_val = integrate(f, (x, a, b))
        plt.plot((s + 1, t), [true_val] * 2, label='true value')
        plt.grid()
        plt.legend()
        plt.title("integration results for " + str(mode))
        plt.xlabel("evaluation points")
        plt.ylabel("integration result")
        plt.show()


def plot_integrations(f, s, t, show_one_by_one=False, a=-1, b=1):
    for mode in Mode:
        plot_integration(f, mode, s, t, show_one_by_one, a, b)
    if not show_one_by_one:
        true_val = integrate(f, (x, a, b))
        plt.plot((s + 1, t), [true_val] * 2, label='true value')
        # plt.scatter((s, t-1), [true_val] * 2 , label = 'true_val')
        plt.title("integration results for all modes")
        plt.xlabel("evaluation points")
        plt.ylabel("integration result")
        plt.grid()
        plt.legend()
        plt.show()
