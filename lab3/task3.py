"""
3.3. Для таблично заданной функции путём решения нормальной системы
МНК найти приближающие многочлены а) 1-й и б) 2-й степени. Для каждого
из приближающих многочленов вычислить сумму квадратов ошибок.
Построить графики приближаемой функции и приближающих многочленов.

Вариант 7.
i      0     1       2       3       4       5
x_i    0.0   0.2     0.4     0.6     0.8     1.0
y_i    1.0   1.0032  1.0512  1.2592  1.8192  3.0
"""

import matplotlib.pyplot as plt


def normal_system(x, y, degree):
    """Строит матрицу и правую часть нормальной системы МНК."""

    # M[k][i] = сумма x_j^(k+i), b[k] = сумма y_j*x_j^k, формула (3.17).
    powers = [sum(xj**k for xj in x) for k in range(2 * degree + 1)]
    matrix = [[powers[k + i]
               for i in range(degree + 1)] for k in range(degree + 1)]
    rhs = [sum(yj * xj**k for xj, yj in zip(x, y)) for k in range(degree + 1)]

    return matrix, rhs


def solve_normal_system(matrix, rhs):
    """Решает нормальную систему методом Гаусса."""

    n = len(rhs)
    rows = [matrix[i] + [rhs[i]] for i in range(n)]
    # При полном ранге матрица МНК положительно определена,
    # поэтому в прямом ходе ведущие элементы должны быть положительными.
    for k in range(n):
        pivot = rows[k][k]
        if pivot <= 0:
            raise ValueError(
                "Нормальная система потеряла положительную определённость.")
        for i in range(k + 1, n):
            multiplier = rows[i][k] / pivot
            rows[i][k] = 0.0
            for j in range(k + 1, n + 1):
                rows[i][j] -= multiplier * rows[k][j]

    a = [0.0] * n
    for i in range(n - 1, -1, -1):
        known = sum(rows[i][j] * a[j] for j in range(i + 1, n))
        a[i] = (rows[i][-1] - known) / rows[i][i]

    return a


def polynomial(a, x):
    """Вычисляет a_0 + a_1*x + ... в точке x."""

    return sum(coefficient * x**i for i, coefficient in enumerate(a))


def show_plots(x, y, coefficients):
    """Показывает табличные точки и оба приближающих многочлена."""

    left, right = min(x), max(x)
    plot_x = [left + i * (right - left) / 200 for i in range(201)]
    _, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.scatter(x, y, color="black", zorder=3, label="Табличные данные")
    for degree, (a, style) in enumerate(zip(coefficients, ("-", "--")), start=1):
        ax.plot(plot_x, [polynomial(a, xi) for xi in plot_x], style,
                label=f"Многочлен степени {degree}")
    ax.set(xlabel="x", ylabel="y",
           title="3.3. Приближение методом наименьших квадратов")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()


def main():
    """Находит оба приближения, печатает проверки и строит графики."""

    x = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y = [1.0, 1.0032, 1.0512, 1.2592, 1.8192, 3.0]
    print("3.3. Вариант 7. Метод наименьших квадратов")
    print(f"x = {x}\ny = {y}")

    coefficients = []
    for degree in (1, 2):
        if len(x) != len(y) or len(set(x)) < degree + 1:
            raise ValueError(
                f"Для степени {degree} нужны минимум {degree + 1} разных узлов.")
        matrix, rhs = normal_system(x, y, degree)
        a = solve_normal_system(matrix, rhs)
        coefficients.append(a)

        error = sum((polynomial(a, xi) - yi)**2 for xi, yi in zip(x, y))
        residual = max(abs(sum(v * aj for v, aj in zip(row, a)) - bi)
                       for row, bi in zip(matrix, rhs))
        formula = f"{a[0]:.8f} {a[1]:+.8f}*x"
        if degree == 2:
            formula += f" {a[2]:+.8f}*x²"
        print(f"\nF_{degree}(x) = {formula}")
        print(f"Сумма квадратов ошибок: {error:.9f}")
        print(f"Максимальная невязка нормальной системы: {residual:.2e}")
    show_plots(x, y, coefficients)


if __name__ == "__main__":
    main()
