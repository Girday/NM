"""
4.2. Реализовать метод стрельбы и конечно-разностный метод решения краевой
задачи для ОДУ в виде программ. С использованием разработанного программного
обеспечения решить краевую задачу для обыкновенного дифференциального
уравнения 2-го порядка на указанном отрезке. Оценить погрешность численного
решения с использованием метода Рунге–Ромберга и путём сравнения с точным
решением.

Вариант 7.
(2x + 1)y'' + 4xy' − 4y = 0.
y'(0) = −1, y'(1) + 2y(1) = 3.
x ∈ [0, 1].
Точное решение y = x + e^(−2x).
"""

from math import exp

from ode import integrate, make_grid, print_errors, runge_kutta_step


def p(x):
    """Возвращает коэффициент при y' в уравнении y'' + p(x)y' + q(x)y = 0."""

    return 4 * x / (2 * x + 1)


def q(x):
    """Возвращает коэффициент при y в уравнении y'' + p(x)y' + q(x)y = 0."""

    return -4 / (2 * x + 1)


def g(x, y, z):
    """Вычисляет правую часть уравнения для z = y'."""

    return -p(x) * z - q(x) * y


def exact_solution(x):
    """Возвращает точное решение варианта 7."""

    return x + exp(-2 * x)


def shooting(x, h, z0, gamma, right_value, guesses, eps, max_iter):
    """Подбирает y(0) методом секущих и возвращает решение и историю проб."""

    if eps <= 0:
        raise ValueError("Точность должна быть положительной.")

    def trial(eta):
        """Решает задачу Коши с y(0) = eta и проверяет правое условие."""

        y, z = integrate(g, x, h, eta, z0, runge_kutta_step)
        phi = z[-1] + gamma * y[-1] - right_value
        return y, z, phi

    eta_prev, eta = guesses
    y, z, phi_prev = trial(eta_prev)
    history = [(eta_prev, phi_prev)]
    if abs(phi_prev) <= eps:
        return y, z, history

    y, z, phi = trial(eta)
    history.append((eta, phi))
    for _ in range(max_iter):
        if abs(phi) <= eps:
            return y, z, history
        difference = phi - phi_prev
        if abs(difference) <= 1e-14:
            raise RuntimeError(
                "Разность невязок слишком мала для метода секущих.")

        # Нуль секущей через точки (eta_prev, phi_prev) и (eta, phi), (4.34).
        eta_next = eta - phi * (eta - eta_prev) / difference
        eta_prev, phi_prev = eta, phi
        eta = eta_next
        y, z, phi = trial(eta)
        history.append((eta, phi))

    if abs(phi) <= eps:
        return y, z, history

    raise RuntimeError("Метод стрельбы не достиг заданной точности.")


def sweep(a, b, c, d):
    """Решает трёхдиагональную систему методом прогонки."""

    n = len(b)
    P = [0.0] * n
    Q = [0.0] * n
    P_prev = Q_prev = 0.0  # В первой строке a[0] = 0.
    for k in range(n):
        # Подставляем y[k-1] = P[k-1]*y[k] + Q[k-1] в k-ю строку.
        denominator = b[k] + a[k] * P_prev
        if abs(denominator) <= 1e-14:
            raise ZeroDivisionError(
                "Слишком малый знаменатель в методе прогонки.")
        P[k] = -c[k] / denominator
        Q[k] = (d[k] - a[k] * Q_prev) / denominator
        P_prev, Q_prev = P[k], Q[k]

    y = [0.0] * n
    y[-1] = Q[-1]  # В последней строке c[-1] = 0.
    for k in range(n - 2, -1, -1):
        y[k] = P[k] * y[k + 1] + Q[k]

    return y


def finite_difference(x, h, z0, gamma, right_value):
    """Составляет и решает разностную систему второго порядка."""

    n = len(x)
    if n < 3:
        raise ValueError(
            "Для граничных формул второго порядка нужны три узла.")

    a, b, c, d = [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * n

    # Центральные разности (4.37). Умножаем строки на h².
    for k in range(1, n - 1):
        a[k] = 1 - h * p(x[k]) / 2
        b[k] = -2 + h**2 * q(x[k])
        c[k] = 1 + h * p(x[k]) / 2

    # Односторонние разности (4.41). Граничные строки умножены на 2h.
    # Убираем -y[2] слева и +y[n-3] справа соседними внутренними строками.
    if abs(c[1]) <= 1e-14 or abs(a[-2]) <= 1e-14:
        raise ZeroDivisionError(
            "Слишком малый коэффициент для исключения на границе.")

    b[0] = -3 + a[1] / c[1]
    c[0] = 4 + b[1] / c[1]
    d[0] = 2 * h * z0
    a[-1] = -4 - b[-2] / a[-2]
    b[-1] = 3 + 2 * h * gamma - c[-2] / a[-2]
    d[-1] = 2 * h * right_value

    margin = min(abs(b[k]) - abs(a[k]) - abs(c[k]) for k in range(n))

    return sweep(a, b, c, d), margin


def difference_residuals(x, h, y, z0, gamma, right_value):
    """Проверяет разностное уравнение и оба условия до исключения коэффициентов."""

    residual = 0.0
    for k in range(1, len(x) - 1):
        dy = (y[k + 1] - y[k - 1]) / (2 * h)
        ddy = (y[k + 1] - 2 * y[k] + y[k - 1]) / h**2
        residual = max(residual, abs(ddy + p(x[k]) * dy + q(x[k]) * y[k]))
    left = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * h) - z0
    right = (3 * y[-1] - 4 * y[-2] + y[-3]) / \
        (2 * h) + gamma * y[-1] - right_value

    return residual, left, right


def main():
    """Решает краевую задачу на двух сетках и проверяет результат."""

    a, b = 0.0, 1.0
    # y'(a), коэффициент при y(b), правая часть.
    z0, gamma, right_value = -1.0, 2.0, 3.0
    h = 0.1  # В условии шаг 4.2 не задан.
    guesses = (0.0, 2.0)  # На этих пробах невязка правого условия меняет знак.
    eps, max_iter = 1e-10, 20
    x = make_grid(a, b, h)
    x_half = make_grid(a, b, h / 2)

    y_shoot, z_shoot, history = shooting(
        x, h, z0, gamma, right_value, guesses, eps, max_iter)
    y_shoot_half, _, history_half = shooting(x_half, h / 2, z0, gamma, right_value,
                                             guesses, eps, max_iter)
    y_diff, margin = finite_difference(x, h, z0, gamma, right_value)
    y_diff_half, margin_half = finite_difference(
        x_half, h / 2, z0, gamma, right_value)
    exact = [exact_solution(point) for point in x]
    exact_half = [exact_solution(point) for point in x_half]

    print("4.2. Вариант 7. (2x + 1)y'' + 4xy' − 4y = 0")
    print(f"[{a:g}, {b:g}], y'({a:g}) = {z0:g}, y'({b:g}) + {gamma:g}y({b:g}) = {right_value:g}, "
          f"h = {h:g}, h/2 = {h/2:g}")
    print(f"2x + 1 ≥ {2*a+1:g}, поэтому p и q непрерывны на всём отрезке.\n")
    print(
        f"Стрельба. Критерий |Φ(η)| ≤ {eps:g}, не более {max_iter} шагов секущих.")
    print("Из-за линейности задачи Коши Φ(η) имеет вид Aη + B.")
    print("При разных невязках у двух проб секущая даёт корень за один шаг.")
    for k, (eta, phi) in enumerate(history):
        print(f"  Проба {k}: η = {eta:.9f}, Φ = {phi:+.3e}")
    print(
        f"Шагов секущих на h и h/2: {max(0, len(history) - 2)} и {max(0, len(history_half) - 2)}")
    print(f"Модули краевых невязок стрельбы на h: {abs(z_shoot[0] - z0):.3e} и "
          f"{abs(history[-1][1]):.3e}")

    ode_residual, left, right = difference_residuals(
        x, h, y_diff, z0, gamma, right_value)
    print(
        f"\nМинимум |bₖ| − |aₖ| − |cₖ| на h и h/2: {margin:.3e} и {margin_half:.3e}")
    print(f"Разности на h. Макс. невязка уравнения: {ode_residual:.3e}, "
          f"модули краевых невязок: {abs(left):.3e} и {abs(right):.3e}\n")

    print(f"{'x':>6} {'Точное y':>12} {'Стрельба':>12} {'Разности':>12}")
    for point, ref, shoot, diff in zip(x, exact, y_shoot, y_diff):
        print(f"{point:6.3f} {ref:12.8f} {shoot:12.8f} {diff:12.8f}")

    print("\nМаксимальные ошибки y. Оценка Рунге–Ромберга для h/2 берётся в общих узлах.")
    print(f"{'Метод':19} {'p':>2}   {'Ошибка h':>12}   {'Ошибка h/2':>12}   {'Оценка h/2':>12}")
    print_errors("Стрельба", 4, y_shoot, y_shoot_half, exact, exact_half)
    print_errors("Конечные разности", 2, y_diff,
                 y_diff_half, exact, exact_half)


if __name__ == "__main__":
    main()
