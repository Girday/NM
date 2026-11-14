"""
2.2.
Реализовать методы простой итерации и Ньютона решения систем нелинейных
уравнений. В качестве входных данных задать точность вычислений.

Найти решение системы, в котором обе неизвестные положительны (начальное
приближение определить графически). Проанализировать зависимость
погрешности вычислений от количества итераций.

Вариант 7.
a = 2.
x1^2 + x2^2 - a^2 = 0
x1 - e^x2 + a = 0
"""

from math import exp, log, sqrt

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def f(x1, x2):
    """Вычисляет левые части двух уравнений при a = 2."""

    return x1**2 + x2**2 - 4, x1 - exp(x2) + 2


def phi(x1, x2):
    """Вычисляет новую пару для простой итерации в первой четверти."""

    return sqrt(4 - x2**2), log(x1 + 2)


def max_difference(x, y):
    """Возвращает наибольшую разность координат по модулю."""

    return max(abs(x[0] - y[0]), abs(x[1] - y[1]))


def simple_iteration(x0, q, eps, max_iter=100):
    """Возвращает приближения простой итерации и последнюю оценку погрешности."""

    x = x0
    history = [x]
    for _ in range(max_iter):
        x1, x2 = x
        x_next = phi(x1, x2)  # Обе новые координаты считаем из старой пары.
        error_bound = q / (1 - q) * max_difference(x_next, x)  # Оценка (2.26).
        history.append(x_next)
        if error_bound <= eps:
            return history, error_bound
        x = x_next

    raise RuntimeError("Метод простой итерации не достиг заданной точности.")


def newton(x0, eps, max_iter=100):
    """Возвращает приближения Ньютона и норму последнего шага."""

    x = x0
    history = [x]
    for _ in range(max_iter):
        x1, x2 = x
        f1, f2 = f(x1, x2)

        # Строки матрицы Якоби — частные производные f1 и f2.
        j11, j12 = 2 * x1, 2 * x2
        j21, j22 = 1, -exp(x2)
        det_j = j11 * j22 - j12 * j21
        if abs(det_j) < 1e-14:
            raise RuntimeError(
                "Определитель матрицы Якоби слишком близок к нулю.")

        # Решаем J * delta = -f по формулам Крамера.
        det_1 = -f1 * j22 + j12 * f2
        det_2 = -j11 * f2 + f1 * j21
        x_next = (x1 + det_1 / det_j, x2 + det_2 / det_j)
        step = max_difference(x_next, x)
        history.append(x_next)
        if step <= eps:
            return history, step
        x = x_next

    raise RuntimeError("Метод Ньютона не достиг заданной точности.")


def print_result(name, x, criterion, errors):
    """Печатает решение, невязки и погрешность на каждом шаге."""

    x1, x2 = x
    f1, f2 = f(x1, x2)
    label, value = criterion
    print(f"\n{name}. Итераций: {len(errors) - 1}")
    print(f"Ответ: ({x1:.15g}, {x2:.15g})")
    print(f"Невязки: {abs(f1):.2e}, {abs(f2):.2e}")
    print(f"{label}: {value:.2e}")
    print("  k   Погрешность")
    for k, error in enumerate(errors):
        print(f"{k:3d}   {error:.2e}")


def show_plots(x0, iteration_errors, newton_errors):
    """Строит кривые системы 2.2 и отмечает область G и начальную точку."""

    _, (curves, errors) = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained")
    xs = [1.3 + i * 0.005 for i in range(101)]
    curves.plot(xs, [sqrt(4 - x**2) for x in xs], label=r"$x_1^2+x_2^2=4$")
    curves.plot(xs, [log(x + 2) for x in xs], label=r"$x_2=\ln(x_1+2)$")
    curves.plot([1.5, 1.6, 1.6, 1.5, 1.5], [1.2, 1.2, 1.3, 1.3, 1.2],
                "k--", label="Область G")
    curves.plot(x0[0], x0[1], "ko", label=f"x⁽⁰⁾ = {x0}")
    curves.set(xlabel="x₁", ylabel="x₂", title="Локализация решения")
    errors.set_yscale("log", nonpositive="mask")  # Нулевые ошибки пропускаем.
    errors.xaxis.set_major_locator(MaxNLocator(integer=True))
    errors.plot(iteration_errors, "o-", label="Простая итерация")
    errors.plot(newton_errors, "o-", label="Ньютон")
    errors.set(xlabel="Номер итерации",
               ylabel="Погрешность", title="Сходимость")
    for ax in (curves, errors):
        ax.grid(alpha=0.3)
        ax.legend()
    plt.show()


def main():
    """Решает систему варианта 7 обоими методами, выводит результаты и показывает графики."""

    eps = 1e-6  # Точность вычислений.
    x0 = (1.55, 1.25)  # Середина G = [1.5, 1.6] × [1.2, 1.3].
    q = max(1.3 / sqrt(4 - 1.3**2), 1 / 3.5)  # Граница нормы phi'.
    # Эталон из скалярного уравнения.
    x_ref = (1.5479914435766522, 1.2663816528256687)

    iteration_history, bound = simple_iteration(x0, q, eps)
    newton_history, step = newton(x0, eps)
    iteration_errors = [max_difference(x, x_ref) for x in iteration_history]
    newton_errors = [max_difference(x, x_ref) for x in newton_history]

    print("2.2. Вариант 7. x1**2 + x2**2 = 4, x1 = exp(x2) - 2")
    print(
        f"G = [1.5, 1.6] × [1.2, 1.3], x0 = {x0}, eps = {eps:g}, q = {q:.6f}")
    print_result("Простая итерация", iteration_history[-1],
                 ("Оценка погрешности", bound), iteration_errors)
    print_result("Ньютон", newton_history[-1],
                 ("Последний шаг", step), newton_errors)
    show_plots(x0, iteration_errors, newton_errors)


if __name__ == "__main__":
    main()
