"""Общие вычисления для задач Коши и краевой задачи."""

from math import isclose


def make_grid(a, b, h):
    """Строит равномерную сетку на отрезке [a, b]."""

    if h <= 0 or b <= a:
        raise ValueError("Нужны положительный шаг и a < b.")

    n = round((b - a) / h)
    if n < 1 or not isclose(n * h, b - a, rel_tol=1e-12):
        raise ValueError("Заданный шаг должен делить длину отрезка.")

    return [a + k * h for k in range(n)] + [b]


def runge_kutta_step(g, x, y, z, h):
    """Делает один шаг Рунге–Кутты 4-го порядка для y' = z, z' = g(x, y, z)."""

    # K и L — приращения для y и z по формуле (4.16).
    K1 = h * z
    L1 = h * g(x, y, z)
    K2 = h * (z + L1 / 2)
    L2 = h * g(x + h / 2, y + K1 / 2, z + L1 / 2)
    K3 = h * (z + L2 / 2)
    L3 = h * g(x + h / 2, y + K2 / 2, z + L2 / 2)
    K4 = h * (z + L3)
    L4 = h * g(x + h, y + K3, z + L3)

    return (y + (K1 + 2 * K2 + 2 * K3 + K4) / 6,
            z + (L1 + 2 * L2 + 2 * L3 + L4) / 6)


def integrate(g, x, h, y0, z0, step):
    """Вычисляет y и z во всех узлах заданным одношаговым методом."""

    y, z = [y0], [z0]
    for k in range(len(x) - 1):
        y_next, z_next = step(g, x[k], y[k], z[k], h)
        y.append(y_next)
        z.append(z_next)

    return y, z


def print_errors(name, order, y, y_half, exact, exact_half):
    """Выводит максимальные ошибки и оценку Рунге–Ромберга для шага h/2."""

    error = max(abs(value - ref) for value, ref in zip(y, exact))
    error_half = max(abs(value - ref)
                     for value, ref in zip(y_half, exact_half))
    # Узел k сетки с шагом h совпадает с узлом 2*k сетки с шагом h/2.
    estimate = max(abs(y_half[2 * k] - y[k]) for k in range(len(y)))
    estimate /= 2**order - 1
    print(f"{name:19} {order:2d}   {error:12.3e}   {error_half:12.3e}   {estimate:12.3e}")
