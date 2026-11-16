"""
3.4. Вычислить первую и вторую производные таблично заданной
функции y_i = f(x_i), i = 0, 1, 2, 3, 4, в точке x = X*.

Вариант 7.
X* = 0.2.
i      0       1       2       3       4
x_i   -0.2     0.0     0.2     0.4     0.6
y_i    1.7722  1.5708  1.3694  1.1593  0.9273
"""

from math import sqrt


def differentiate(x, y, i):
    """Дифференцирует квадратичный многочлен по трём узлам вокруг x[i]."""

    if len(x) != len(y) or not 0 < i < len(x) - 1:
        raise ValueError(
            "Нужны значения функции в точке и в двух соседних узлах.")

    h_left = x[i] - x[i - 1]
    h_right = x[i + 1] - x[i]
    if min(h_left, h_right) <= 0:
        raise ValueError("Соседние узлы должны строго возрастать.")

    # Формула (3.18) даёт наклоны двух соседних секущих.
    left = (y[i] - y[i - 1]) / h_left
    right = (y[i + 1] - y[i]) / h_right
    second_difference = (right - left) / (h_left + h_right)

    # Парабола в форме Ньютона: y[i-1] + left*(x - x[i-1])
    # + second_difference*(x - x[i-1])*(x - x[i]).
    # Подставляем x[i] в её производные по формулам (3.20) и (3.21).
    first = left + second_difference * h_left
    second = 2 * second_difference

    return left, right, first, second


def main():
    """Считает производные по таблице и сравнивает их с эталоном."""

    x = [-0.2, 0.0, 0.2, 0.4, 0.6]
    y = [1.7722, 1.5708, 1.3694, 1.1593, 0.9273]
    point = 0.2
    i = x.index(point)
    left, right, first, second = differentiate(x, y, i)

    # Табличные значения совпадают с округлёнными значениями arccos(x).
    # Эталон нужен только для сравнения, в разностях используются числа из условия.
    exact_first = -1 / sqrt(1 - point**2)
    exact_second = -point / (1 - point**2)**1.5

    print(f"3.4. Вариант 7. Производные в точке X* = {point:g}")
    print(f"Узлы для расчёта: {x[i-1:i+2]}")
    print(f"Наклон левой секущей: {left:.6f}, правой: {right:.6f}")
    print(f"f'({point:g}) ≈ {first:.6f}")
    print(f"f''({point:g}) ≈ {second:.6f}")
    print(
        f"Эталон arccos(x): f' = {exact_first:.6f}, f'' = {exact_second:.6f}")
    print("Абсолютные ошибки по сравнению с arccos(x): "
          f"{abs(first - exact_first):.3e}, {abs(second - exact_second):.3e}")


if __name__ == "__main__":
    main()
