"""
1.1. Реализовать алгоритм LU-разложения матриц (с выбором главного элемента).
Решить СЛАУ, вычислить определитель и обратную матрицу.

Вариант №7
 1  -5  -7   1 | -75
 1  -3  -9  -4 | -41
-2   4   2   1 |  18
-9   9   5   3 |  29
"""

import numpy as np

A = [
    [1, -5, -7, 1],
    [1, -3, -9, -4],
    [-2, 4, 2, 1],
    [-9, 9, 5, 3],
]
b = [-75, -41, 18, 29]


def lu_decomposition(matrix):
    """Находит матрицы L, U и перестановки строк."""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Матрица должна быть квадратной")

    # U сначала равна A, а L сначала является единичной матрицей
    U = [[float(value) for value in row] for row in matrix]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0

    # p хранит новый порядок строк матрицы A
    p = list(range(n))
    num_swaps = 0

    # Порог нужен, чтобы не делить на ноль или очень маленькое число
    scale = max(abs(value) for row in U for value in row)
    eps = 1e-12 * max(1.0, scale)

    for k in range(n):
        # Ищем максимальный по модулю элемент в столбце k
        max_index = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > abs(U[max_index][k]):
                max_index = i

        if abs(U[max_index][k]) < eps:
            raise ValueError(
                f"Главный элемент в столбце {k + 1} равен нулю "
                "или слишком мал"
            )

        # Ставим строку с главным элементом на место строки k
        if max_index != k:
            U[k], U[max_index] = U[max_index], U[k]
            p[k], p[max_index] = p[max_index], p[k]

            # В L меняем только уже найденные коэффициенты
            for j in range(k):
                L[k][j], L[max_index][j] = L[max_index][j], L[k][j]

            num_swaps += 1

        # Обнуляем элементы под главным элементом
        for i in range(k + 1, n):
            mu = U[i][k] / U[k][k]
            L[i][k] = mu

            for j in range(k, n):
                U[i][j] -= mu * U[k][j]

            U[i][k] = 0.0

    return L, U, p, num_swaps


def find_x(L, U, right_side):
    """Решает системы Lz = right_side и Ux = z."""
    n = len(right_side)

    # Прямая подстановка: находим z из Lz = right_side
    z = [0.0] * n
    for i in range(n):
        z_sum = 0.0
        for j in range(i):
            z_sum += L[i][j] * z[j]
        z[i] = (right_side[i] - z_sum) / L[i][i]

    # Обратная подстановка: находим x из Ux = z
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x_sum = 0.0
        for j in range(i + 1, n):
            x_sum += U[i][j] * x[j]
        x[i] = (z[i] - x_sum) / U[i][i]

    return x


def find_determinant(U, num_swaps):
    """Вычисляет определитель по диагонали U."""
    det = (-1) ** num_swaps
    for i in range(len(U)):
        det *= U[i][i]
    return det


def find_inverse(L, U, p):
    """Находит обратную матрицу по одному столбцу."""
    n = len(L)
    A_inv = [[0.0] * n for _ in range(n)]

    for col in range(n):
        # e — столбец единичной матрицы
        e = [0.0] * n
        e[col] = 1.0

        # Решаем Ax = e, то есть LUx = Pe
        e_mod = [e[p[i]] for i in range(n)]
        x_col = find_x(L, U, e_mod)

        for i in range(n):
            A_inv[i][col] = x_col[i]

    return A_inv


def make_permutation_matrix(p):
    """Создаёт матрицу P по вектору перестановок."""
    n = len(p)
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        P[i][p[i]] = 1.0
    return P


def main():
    L, U, p, num_swaps = lu_decomposition(A)

    # Из PA = LU следует LUx = Pb
    b_mod = [b[p[i]] for i in range(len(b))]
    x = find_x(L, U, b_mod)

    det = find_determinant(U, num_swaps)
    A_inv = find_inverse(L, U, p)
    P = make_permutation_matrix(p)

    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    L_np = np.array(L)
    U_np = np.array(U)
    P_np = np.array(P)
    x_np = np.array(x)
    A_inv_np = np.array(A_inv)

    PA = P_np @ A_np
    LU = L_np @ U_np
    Ax = A_np @ x_np
    A_A_inv = A_np @ A_inv_np
    identity = np.eye(len(A))

    np.set_printoptions(precision=6, suppress=True)
    print("Порядок строк после перестановок:", [index + 1 for index in p])
    print("Количество перестановок:", num_swaps)
    print("\nМатрица P:\n", P_np, sep="")
    print("\nМатрица L:\n", L_np, sep="")
    print("\nМатрица U:\n", U_np, sep="")
    print("\nРешение x:", x_np)
    print("Определитель:", det)
    print("\nОбратная матрица:\n", A_inv_np, sep="")

    print("\nПроверка P @ A = L @ U:")
    print("P @ A =\n", PA, sep="")
    print("L @ U =\n", LU, sep="")
    print("Равенство выполняется:", np.allclose(PA, LU))

    print("\nПроверка A @ x = b:")
    print("A @ x =", Ax)
    print("b     =", b_np)
    print("Равенство выполняется:", np.allclose(Ax, b_np))

    print("\nПроверка A @ A_inv = I:")
    print("A @ A_inv =\n", A_A_inv, sep="")
    print("I =\n", identity, sep="")
    print("Равенство выполняется:", np.allclose(A_A_inv, identity))


if __name__ == "__main__":
    main()
