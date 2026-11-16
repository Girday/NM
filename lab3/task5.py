"""
3.5. Вычислить определённый интеграл F = ∫ y dx от X_0 до X_k
методами прямоугольников, трапеций, Симпсона с шагами h_1, h_2.
Оценить погрешность вычислений, используя метод Рунге–Ромберга.

Вариант 7.
y = 1 / (3*x^2 + 4*x + 2).
X_0 = -2, X_k = 2, h_1 = 1.0, h_2 = 0.5.
"""

from math import atan, isclose, sqrt


def f(x):
    """Вычисляет подынтегральную функцию варианта 7."""

    return 1 / (3 * x**2 + 4 * x + 2)


def antiderivative(x):
    """Вычисляет первообразную подынтегральной функции для проверки."""

    # 3*x² + 4*x + 2 = ((3*x + 2)² + 2) / 3.
    return atan((3 * x + 2) / sqrt(2)) / sqrt(2)


def make_grid(a, b, h):
    """Строит равномерную сетку с заданным шагом на всём отрезке."""

    if h <= 0 or b <= a:
        raise ValueError("Нужны положительный шаг и a < b.")

    n = round((b - a) / h)
    if n == 0 or not isclose(n * h, b - a, rel_tol=1e-12):
        raise ValueError("Заданный шаг должен делить длину отрезка.")

    return [a + i * h for i in range(n)] + [b]


def rectangles(function, x, h):
    """Вычисляет интеграл по формуле средних прямоугольников."""

    total = 0.0
    for i in range(len(x) - 1):
        total += function((x[i] + x[i + 1]) / 2)

    return h * total


def trapezoids(function, x, h):
    """Вычисляет интеграл по составной формуле трапеций."""

    total = (function(x[0]) + function(x[-1])) / 2
    for i in range(1, len(x) - 1):
        total += function(x[i])

    return h * total


def simpson(function, x, h):
    """Вычисляет интеграл по составной формуле Симпсона."""

    n = len(x) - 1
    if n < 2 or n % 2 != 0:
        raise ValueError(
            "Методу Симпсона нужно положительное чётное число интервалов.")

    total = function(x[0]) + function(x[-1])
    for i in range(1, n):
        # Нечётные узлы лежат посередине пар отрезков,
        # чётные внутренние узлы общие для двух соседних пар.
        weight = 4 if i % 2 != 0 else 2
        total += weight * function(x[i])

    return h / 3 * total


def runge_romberg(coarse, fine, k, p):
    """Возвращает поправку к результату на мелкой сетке и уточнённый интеграл."""

    # Разность приближений устраняет неизвестный коэффициент при h^p, (3.30).
    correction = (fine - coarse) / (k**p - 1)

    return correction, fine + correction


def main():
    """Сравнивает три метода на заданных шагах с аналитическим результатом."""

    a, b = -2.0, 2.0
    h1, h2 = 1.0, 0.5
    x1 = make_grid(a, b, h1)
    x2 = make_grid(a, b, h2)
    k = h1 / h2
    if k <= 1:
        raise ValueError("Для уточнения нужен шаг h1 > h2.")

    exact = antiderivative(b) - antiderivative(a)
    print("3.5. Вариант 7. f(x) = 1 / (3*x² + 4*x + 2)")
    print(f"[{a:g}, {b:g}], h1 = {h1:g} ({len(x1)-1} интервала), "
          f"h2 = {h2:g} ({len(x2)-1} интервалов)")
    print("Знаменатель равен 3*(x + 2/3)² + 2/3 и положителен на всём отрезке.")
    print(f"Точное значение интеграла: {exact:.9f}\n")

    methods = [("Прямоугольники", rectangles, 2), ("Трапеции", trapezoids, 2),
               ("Симпсон", simpson, 4)]
    print(f"{'Метод':15} {'p':>2} {'F(h1)':>12} {'F(h2)':>12} {'Оценка h2':>12} {'Уточнение':>12}")
    for name, method, p in methods:
        coarse = method(f, x1, h1)
        fine = method(f, x2, h2)
        correction, refined = runge_romberg(coarse, fine, k, p)
        print(
            f"{name:15} {p:2d} {coarse:12.9f} {fine:12.9f} {abs(correction):12.3e} {refined:12.9f}")
        print(f"Абсолютные ошибки h1, h2 и после уточнения: "
              f"{abs(coarse - exact):.3e}, {abs(fine - exact):.3e}, {abs(refined - exact):.3e}")


if __name__ == "__main__":
    main()
