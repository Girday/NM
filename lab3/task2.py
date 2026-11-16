"""
3.2. Построить кубический сплайн для функции, заданной в узлах
интерполяции, предполагая, что сплайн имеет нулевую кривизну
при x = x_0 и x = x_4. Вычислить значение функции в точке x = X*.

Вариант 7.
X* = 3.0.
i      0       1       2       3       4
x_i    0.0     1.7     3.4     5.1     6.8
f_i    0.0     1.3038  1.8439  2.2583  2.6077
"""


def spline_coefficients(x, f):
    """Возвращает коэффициенты (a_i, b_i, c_i, d_i) на каждом отрезке."""

    if len(x) != len(f) or len(x) < 2:
        raise ValueError("Нужно не менее двух узлов со значениями функции.")

    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    if min(h) <= 0:
        raise ValueError("Узлы сплайна должны строго возрастать.")
    slopes = [(f[i + 1] - f[i]) / h[i] for i in range(len(h))]

    # c[j] = S''(x[j]) / 2. На концах отрезка эти значения равны нулю.
    c = [0.0] * len(x)
    P = [0.0] * len(x)
    Q = [0.0] * len(x)
    for j in range(1, len(x) - 1):
        # Прямой ход по (3.13). Подставляем c[j-1] = P[j-1]*c[j] + Q[j-1].
        # Диагональ 2*(h[j-1]+h[j]) больше суммы двух соседних коэффициентов.
        denominator = 2 * (h[j - 1] + h[j]) + h[j - 1] * P[j - 1]
        if denominator <= 0:
            raise ValueError(
                "Знаменатель прямого хода прогонки должен быть положительным.")
        rhs = 3 * (slopes[j] - slopes[j - 1])
        P[j] = -h[j] / denominator
        Q[j] = (rhs - h[j - 1] * Q[j - 1]) / denominator

    for j in range(len(x) - 2, 0, -1):
        c[j] = P[j] * c[j + 1] + Q[j]

    coefficients = []
    for i in range(len(h)):
        # Формулы (3.14) для отрезка [x[i], x[i+1]] с началом в x[i].
        a = f[i]
        b = slopes[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d = (c[i + 1] - c[i]) / (3 * h[i])
        coefficients.append((a, b, c[i], d))

    return coefficients


def evaluate_piece(coefficients, t):
    """Вычисляет значение части сплайна на расстоянии t от левого узла."""

    a, b, c, d = coefficients

    return a + b * t + c * t**2 + d * t**3


def evaluate_spline(x, coefficients, point):
    """Находит нужный отрезок и вычисляет значение сплайна."""

    for i in range(len(coefficients)):
        if x[i] <= point <= x[i + 1]:
            return evaluate_piece(coefficients[i], point - x[i])

    raise ValueError("Точка лежит за пределами отрезка интерполяции.")


def print_check(x, f, coefficients):
    """Проверяет совпадение с таблицей, стыки и нулевую кривизну на концах."""

    node_error = first_jump = second_jump = 0.0
    first_right = second_right = 0
    for i, (a, b, c, d) in enumerate(coefficients):
        h = x[i + 1] - x[i]
        right_value = evaluate_piece(coefficients[i], h)
        node_error = max(node_error, abs(
            a - f[i]), abs(right_value - f[i + 1]))
        first_right = b + 2 * c * h + 3 * d * h**2
        second_right = 2 * c + 6 * d * h
        if i + 1 < len(coefficients):
            _, next_b, next_c, _ = coefficients[i + 1]
            first_jump = max(first_jump, abs(first_right - next_b))
            second_jump = max(second_jump, abs(second_right - 2 * next_c))

    print(f"Максимальное отклонение в узлах: {node_error:.2e}")
    print(
        f"Расхождение S' и S'' на стыках: {first_jump:.2e}, {second_jump:.2e}")
    print(f"S'' на концах: {2 * coefficients[0][2]:.2e}, {second_right:.2e}")


def main():
    """Строит, выводит и проверяет сплайн варианта 7."""

    x = [0.0, 1.7, 3.4, 5.1, 6.8]
    f = [0.0, 1.3038, 1.8439, 2.2583, 2.6077]
    point = 3.0
    coefficients = spline_coefficients(x, f)

    print("3.2. Вариант 7. Естественный кубический сплайн")
    print("S_i(x) = a_i + b_i*t + c_i*t² + d_i*t³, t = x − x_{i−1}")
    print(
        f"  i   {'Отрезок':>10} {'a_i':>12} {'b_i':>12} {'c_i':>12} {'d_i':>12}")
    for i, (a, b, c, d) in enumerate(coefficients):
        interval = f"[{x[i]:g}, {x[i + 1]:g}]"
        print(f"{i + 1:3d}   {interval:>10} {a:12.8f} {b:12.8f} {c:12.8f} {d:12.8f}")
    print(f"\nS({point:g}) = {evaluate_spline(x, coefficients, point):.9f}")
    print_check(x, f, coefficients)


if __name__ == "__main__":
    main()
