"""
2.1.
Реализовать методы  простой итерации и Ньютона решения нелинейных уравнений.
В качестве  входных  данных задать точность  вычислений.

Найти положительный корень нелинейного  уравнения  (начальное  приближение
определить  графически). Проанализировать зависимость погрешности вычислений
от количества итераций.

Вариант 7.
2^x + x^2 - 2 = 0
"""

from math import log

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def f(x):
    """Вычисляет левую часть уравнения."""

    return 2**x + x**2 - 2


def df(x):
    """Вычисляет производную f'(x) для метода Ньютона."""

    return log(2) * 2**x + 2 * x


def simple_iteration(x0, lam, q, eps, max_iter=100):
    """Возвращает приближения простой итерации и последнюю оценку погрешности."""

    x = x0
    history = [x]
    for _ in range(max_iter):
        x_next = x - lam * f(x)  # Следующее приближение x = phi(x).
        error_bound = q / (1 - q) * abs(x_next - x)  # Оценка (2.7).
        history.append(x_next)
        if error_bound <= eps:
            return history, error_bound
        x = x_next

    raise RuntimeError("Метод простой итерации не достиг заданной точности.")


def newton(x0, eps, max_iter=100):
    """Возвращает приближения Ньютона и модуль последнего шага."""

    x = x0
    history = [x]
    for _ in range(max_iter):
        derivative = df(x)
        if abs(derivative) < 1e-14:
            raise RuntimeError("Производная слишком близка к нулю.")
        # Пересечение касательной с осью x, (2.2).
        x_next = x - f(x) / derivative
        step = abs(x_next - x)
        history.append(x_next)
        if step <= eps:
            return history, step
        x = x_next

    raise RuntimeError("Метод Ньютона не достиг заданной точности.")


def print_result(name, x, criterion, errors):
    """Печатает корень, невязку и погрешность на каждом шаге."""

    label, value = criterion
    print(f"\n{name}. Итераций: {len(errors) - 1}")
    print(f"Корень: {x:.15g}, невязка: {abs(f(x)):.2e}")
    print(f"{label}: {value:.2e}")
    print("  k   Погрешность")
    for k, error in enumerate(errors):
        print(f"{k:3d}   {error:.2e}")


def show_plots(a, b, x0, iteration_errors, newton_errors):
    """Строит графики двух частей уравнения 2.1 и отмечает начальную точку."""

    _, (curves, errors) = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained")
    xs = [0.4 + i * 0.005 for i in range(101)]
    curves.plot(xs, [2**x for x in xs], label=r"$2^x$")
    curves.plot(xs, [2 - x**2 for x in xs], label=r"$2-x^2$")
    curves.axvspan(a, b, alpha=0.12, label=f"[{a}, {b}]")
    curves.axvline(x0, color="gray", linestyle="--", label=f"x₀ = {x0}")
    curves.set(xlabel="x", ylabel="y", title="Локализация корня")
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
    """Решает уравнение варианта 7 обоими методами, выводит результаты и показывает графики."""

    eps = 1e-6  # Точность вычислений.
    a, b = 0.6, 0.7  # Отрезок, на котором пересекаются графики.
    x0 = b  # f(b) > 0 и f''(b) > 0, поэтому Ньютон стартует с правого конца.
    lam = 1 / df(b)
    q = 1 - lam * df(a)  # f' растёт, поэтому 0 <= phi'(x) <= q.
    x_ref = 0.6534825247841734  # Эталон, найденный делением пополам.

    iteration_history, bound = simple_iteration(x0, lam, q, eps)
    newton_history, step = newton(x0, eps)
    iteration_errors = [abs(x - x_ref) for x in iteration_history]
    newton_errors = [abs(x - x_ref) for x in newton_history]

    print("2.1. Вариант 7. 2**x + x**2 - 2 = 0")
    print(f"[{a}, {b}], x0 = {x0}, eps = {eps:g}, lambda = {lam:.6f}, q = {q:.6f}")
    print_result("Простая итерация", iteration_history[-1],
                 ("Оценка погрешности", bound), iteration_errors)
    print_result("Ньютон", newton_history[-1],
                 ("Последний шаг", step), newton_errors)
    show_plots(a, b, x0, iteration_errors, newton_errors)


if __name__ == "__main__":
    main()
