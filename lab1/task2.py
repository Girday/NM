"""
1.2. Реализовать метод прогонки в виде программы, задавая в качестве
входных данных ненулевые элементы матрицы системы и вектор правых частей.
Используя разработанное программное обеспечение, решить СЛАУ с
трёхдиагональной матрицей.

Вариант 7
15  8   0   0   0 |  92
 2 -15  4   0   0 | -84
 0  4  11   5   0 | -77
 0  0  -3  16  -7 |  15
 0  0   0   3   8 | -11
"""


EPS = 1e-12


def check_input(a, b, c, d):
    """Проверяет размеры диагоналей и положение граничных нулей."""
    n = len(b)
    if n == 0:
        raise ValueError("Система не должна быть пустой.")
    if len(a) != n or len(c) != n or len(d) != n:
        raise ValueError("Списки a, b, c и d должны иметь одинаковую длину.")
    if abs(a[0]) > EPS:
        raise ValueError("Первый коэффициент a[0] должен быть равен нулю.")
    if abs(c[-1]) > EPS:
        raise ValueError(
            "Последний коэффициент c[n - 1] должен быть равен нулю.")


def check_stability(a, b, c):
    """Проверяет достаточное условие устойчивости (1.8)."""
    strict_inequality = False

    for i in range(len(b)):
        left = abs(b[i])
        right = abs(a[i]) + abs(c[i])

        if left < right:
            return False
        if left > right:
            strict_inequality = True

    for i in range(1, len(b) - 1):
        if abs(a[i]) <= EPS or abs(c[i]) <= EPS:
            return False

    return strict_inequality


def check_denominator(denominator, i):
    """Не допускает деления на нулевой или слишком малый знаменатель."""
    if abs(denominator) <= EPS:
        raise ZeroDivisionError(
            f"Знаменатель для строки {i + 1} равен нулю или слишком мал."
        )


def forward_sweep(a, b, c, d):
    """Прямой ход: вычисляет прогоночные коэффициенты P и Q."""
    n = len(b)
    P = [0.0] * n
    Q = [0.0] * n

    # Формула (1.5): коэффициенты для первого уравнения.
    denominator = b[0]
    check_denominator(denominator, 0)
    P[0] = -c[0] / denominator
    Q[0] = d[0] / denominator

    # Формула (1.4): коэффициенты для внутренних уравнений.
    for i in range(1, n - 1):
        denominator = b[i] + a[i] * P[i - 1]
        check_denominator(denominator, i)
        P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator

    # Формула (1.6): P_n = 0, так как c_n = 0.
    if n > 1:
        i = n - 1
        denominator = b[i] + a[i] * P[i - 1]
        check_denominator(denominator, i)
        P[i] = 0.0
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator

    return P, Q


def backward_sweep(P, Q):
    """Обратный ход: вычисляет неизвестные x по формуле (1.7)."""
    n = len(Q)
    x = [0.0] * n
    x[n - 1] = Q[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def tridiagonal_solve(a, b, c, d):
    """Решает трёхдиагональную СЛАУ методом правой прогонки."""
    check_input(a, b, c, d)
    P, Q = forward_sweep(a, b, c, d)
    x = backward_sweep(P, Q)
    return P, Q, x


def calculate_residual(a, b, c, d, x):
    """Вычисляет невязку Ax - d по трём диагоналям."""
    residual = []

    for i in range(len(b)):
        left_side = b[i] * x[i]
        if i > 0:
            left_side += a[i] * x[i - 1]
        if i < len(b) - 1:
            left_side += c[i] * x[i + 1]
        residual.append(left_side - d[i])

    return residual


def independent_check(a, b, c, d):
    """Независимая проверка обычным методом Гаусса с выбором главного элемента."""
    n = len(b)
    A = [[0.0] * n for _ in range(n)]

    for i in range(n):
        A[i][i] = b[i]
        if i > 0:
            A[i][i - 1] = a[i]
        if i < n - 1:
            A[i][i + 1] = c[i]

    augmented_matrix = [A[i] + [float(d[i])] for i in range(n)]

    # Прямой ход метода Гаусса для полной матрицы.
    for column in range(n):
        pivot_row = column
        for row in range(column + 1, n):
            if abs(augmented_matrix[row][column]) > abs(
                augmented_matrix[pivot_row][column]
            ):
                pivot_row = row

        pivot = augmented_matrix[pivot_row][column]
        check_denominator(pivot, column)
        augmented_matrix[column], augmented_matrix[pivot_row] = (
            augmented_matrix[pivot_row],
            augmented_matrix[column],
        )

        for row in range(column + 1, n):
            factor = augmented_matrix[row][column] / \
                augmented_matrix[column][column]
            for j in range(column, n + 1):
                augmented_matrix[row][j] -= factor * \
                    augmented_matrix[column][j]

    # Обратный ход метода Гаусса.
    x_check = [0.0] * n
    for i in range(n - 1, -1, -1):
        right_side = augmented_matrix[i][n]
        for j in range(i + 1, n):
            right_side -= augmented_matrix[i][j] * x_check[j]
        x_check[i] = right_side / augmented_matrix[i][i]

    return x_check


def print_vector(name, values):
    rounded_values = [round(value, 10) for value in values]
    print(f"{name} = {rounded_values}")


def main():
    a = [0, 2, 4, -3, 3]
    b = [15, -15, 11, 16, 8]
    c = [8, 4, 5, -7, 0]
    d = [92, -84, -77, 15, -11]

    check_input(a, b, c, d)
    if check_stability(a, b, c):
        print("Достаточное условие устойчивости (1.8) выполнено.")
    else:
        print("Достаточное условие устойчивости (1.8) не выполнено.")

    P, Q, x = tridiagonal_solve(a, b, c, d)
    residual = calculate_residual(a, b, c, d, x)
    x_check = independent_check(a, b, c, d)
    max_difference = max(abs(x[i] - x_check[i]) for i in range(len(x)))

    print_vector("P", P)
    print_vector("Q", Q)
    print_vector("x", x)
    print_vector("Невязка Ax - d", residual)
    print_vector("Независимая проверка", x_check)
    print(f"Максимальное отличие решений: {max_difference:.3e}")


if __name__ == "__main__":
    main()
