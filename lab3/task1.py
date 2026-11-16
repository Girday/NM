"""
3.1. Используя таблицу значений Y_i функции y = f(x), вычисленных
в точках X_i, i = 0, ..., 3, построить интерполяционные многочлены
Лагранжа и Ньютона, проходящие через точки (X_i, Y_i).
Вычислить значение погрешности интерполяции в точке X*.

Вариант 7.
y = sqrt(x), X* = 3.0.
а) X_i = 0, 1.7, 3.4, 5.1.
б) X_i = 0, 1.7, 4.0, 5.1.
"""

from math import prod, sqrt


def lagrange_coefficients(x, y):
    """Находит коэффициенты при произведениях в форме Лагранжа."""

    coefficients = []
    for i in range(len(x)):
        # В узле x[i] множители сокращаются со знаменателем и дают y[i].
        denominator = prod(x[i] - x[j] for j in range(len(x)) if j != i)
        coefficients.append(y[i] / denominator)

    return coefficients


def lagrange(x, coefficients, point):
    """Вычисляет значение многочлена Лагранжа в заданной точке."""

    value = 0.0
    for i in range(len(x)):
        factors = prod(point - x[j] for j in range(len(x)) if j != i)
        value += coefficients[i] * factors

    return value


def newton_coefficients(x, y):
    """Возвращает первые элементы столбцов таблицы разделённых разностей."""

    differences = y.copy()
    coefficients = [differences[0]]
    for order in range(1, len(x)):
        # Соседние разности предыдущего порядка имеют общие внутренние узлы.
        for i in range(len(x) - order):
            differences[i] = (differences[i + 1] -
                              differences[i]) / (x[i + order] - x[i])
        coefficients.append(differences[0])

    return coefficients


def newton(x, coefficients, point):
    """Вычисляет значение многочлена Ньютона в заданной точке."""

    value = coefficients[0]
    factors = 1.0
    for i in range(1, len(x)):
        factors *= point - x[i - 1]
        value += coefficients[i] * factors

    return value


def print_result(x, y, point, exact, lagrange_coeffs, newton_coeffs):
    """Печатает обе формы многочлена, значения и проверки в узлах."""

    print(f"Узлы: {x}")
    print("Значения y:", "  ".join(f"{value:.8f}" for value in y))
    factors = ["x" if node == 0 else f"(x {(-node):+g})" for node in x]
    lagrange_terms = []
    newton_terms = []
    for i in range(len(x)):
        if lagrange_coeffs[i] != 0:
            lagrange_factors = "*".join(factors[:i] + factors[i + 1:])
            lagrange_terms.append(
                f"{lagrange_coeffs[i]:+.9f}*{lagrange_factors}")
        if i > 0:
            newton_factors = "*".join(factors[:i])
            newton_terms.append(f"{newton_coeffs[i]:+.9f}*{newton_factors}")

    print("L(x) =", " ".join(lagrange_terms) or "0")
    print(f"P(x) = {newton_coeffs[0]:.9f}", " ".join(newton_terms))
    for name, polynomial, coefficients in (
        ("L", lagrange, lagrange_coeffs), ("P", newton, newton_coeffs)
    ):
        value = polynomial(x, coefficients, point)
        node_error = max(abs(polynomial(x, coefficients, xi) - yi)
                         for xi, yi in zip(x, y))
        print(
            f"{name}({point:g}) = {value:.9f}, |f − {name}| = {abs(exact - value):.3e}")
        print(
            f"Максимальное отклонение {name} от таблицы в узлах: {node_error:.2e}")


def main():
    """Строит многочлены на обеих сетках варианта 7."""

    point = 3.0
    exact = sqrt(point)
    node_sets = [("а", [0.0, 1.7, 3.4, 5.1]), ("б", [0.0, 1.7, 4.0, 5.1])]

    print(f"3.1. Вариант 7. f(x) = sqrt(x), f({point:g}) = {exact:.9f}")
    for label, x in node_sets:
        if min(x) < 0 or len(set(x)) != len(x) or not min(x) <= point <= max(x):
            raise ValueError(
                "Нужны различные неотрицательные узлы и X* внутри их диапазона.")
        y = [sqrt(xi) for xi in x]
        print(f"\nНабор {label})")
        print_result(x, y, point, exact, lagrange_coefficients(
            x, y), newton_coefficients(x, y))


if __name__ == "__main__":
    main()
