"""
4.1. Реализовать методы Эйлера, Рунге–Кутты и Адамса 4-го порядка
в виде программ, задавая в качестве входных данных шаг сетки h.
С использованием разработанного программного обеспечения решить задачу
Коши для ОДУ 2-го порядка на указанном отрезке. Оценить погрешность
численного решения с использованием метода Рунге–Ромберга и путём
сравнения с точным решением.

Вариант 7.
y'' − 4xy' + (4x² − 2)y = 0.
y(0) = 1, y'(0) = 1.
x ∈ [0, 1], h = 0.1.
Точное решение y = (1 + x)e^(x²).
"""

from math import exp

from ode import integrate, make_grid, print_errors, runge_kutta_step


def g(x, y, z):
    """Вычисляет правую часть уравнения для z = y'."""

    return 4 * x * z + (2 - 4 * x**2) * y


def exact_solution(x):
    """Возвращает точное решение варианта 7."""

    return (1 + x) * exp(x**2)


def euler_step(g, x, y, z, h):
    """Делает шаг явного Эйлера по наклонам в текущем узле."""

    return y + h * z, z + h * g(x, y, z)


def adams(g, x, h, y0, z0):
    """Строит решение явным методом Адамса 4-го порядка."""

    if len(x) < 4:
        raise ValueError("Для метода Адамса нужны как минимум четыре узла.")

    # Формуле (4.25) нужны значения в x_0, x_1, x_2 и x_3.
    y, z = integrate(g, x[:4], h, y0, z0, runge_kutta_step)
    f = [g(x[k], y[k], z[k]) for k in range(4)]

    for k in range(3, len(x) - 1):
        # Одни и те же веса применяем к y' = z и к z' = g(x, y, z).
        y.append(y[k] + h / 24 * (55 * z[k] - 59 * z[k - 1]
                                  + 37 * z[k - 2] - 9 * z[k - 3]))
        z.append(z[k] + h / 24 * (55 * f[k] - 59 * f[k - 1]
                                  + 37 * f[k - 2] - 9 * f[k - 3]))
        f.append(g(x[k + 1], y[k + 1], z[k + 1]))

    return y, z


def solve(g, x, h, y0, z0):
    """Возвращает значения y для трёх методов на одной сетке."""

    y_euler, _ = integrate(g, x, h, y0, z0, euler_step)
    y_rk, _ = integrate(g, x, h, y0, z0, runge_kutta_step)
    y_adams, _ = adams(g, x, h, y0, z0)

    return y_euler, y_rk, y_adams


def main():
    """Решает задачу Коши на двух сетках и сравнивает погрешности."""

    a, b = 0.0, 1.0
    y0, z0 = 1.0, 1.0
    h = 0.1
    x = make_grid(a, b, h)
    x_half = make_grid(a, b, h / 2)
    solutions = solve(g, x, h, y0, z0)
    solutions_half = solve(g, x_half, h / 2, y0, z0)
    exact = [exact_solution(point) for point in x]
    exact_half = [exact_solution(point) for point in x_half]

    print("4.1. Вариант 7. y'' − 4xy' + (4x² − 2)y = 0")
    print(f"[{a:g}, {b:g}], y({a:g}) = {y0:g}, y'({a:g}) = {z0:g}, "
          f"h = {h:g}, h/2 = {h/2:g}")
    print("Правая часть системы гладкая и линейна по y и z на всём отрезке.\n")
    print(f"{'x':>6} {'Точное y':>12} {'Эйлер':>12} {'Рунге–Кутта':>12} {'Адамс':>12}")
    for point, ref, y_euler, y_rk, y_adams in zip(x, exact, *solutions):
        print(f"{point:6.3f} {ref:12.8f} {y_euler:12.8f} {y_rk:12.8f} {y_adams:12.8f}")

    print("\nМаксимальные ошибки y. Оценка Рунге–Ромберга для h/2 берётся в общих узлах.")
    print(f"{'Метод':19} {'p':>2}   {'Ошибка h':>12}   {'Ошибка h/2':>12}   {'Оценка h/2':>12}")
    methods = [("Эйлер", 1), ("Рунге–Кутта", 4), ("Адамс", 4)]
    for (name, order), y, y_half in zip(methods, solutions, solutions_half):
        print_errors(name, order, y, y_half, exact, exact_half)


if __name__ == "__main__":
    main()
