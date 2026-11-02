'''
1.4. Реализовать метод вращений в виде программы, задавая в качестве входных данных 
матрицу и точность вычислений. Используя разработанное программное обеспечение, найти собственные 
значения и собственные векторы симметрических матриц. 
Проанализировать зависимость погрешности вычислений от числа итераций. 

№7
-6 6 -8
6 -4 9
-8 9 -2
'''

from math import atan, cos, isfinite, pi, sin, sqrt

import numpy as np


def validate_inputs(A, eps, max_iter):
    n = len(A)
    if n == 0:
        raise ValueError("Матрица не должна быть пустой.")
    for row in A:
        if len(row) != n:
            raise ValueError("Матрица должна быть квадратной.")
        for value in row:
            if not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(
                    "Элементы матрицы должны быть конечными числами.")
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][j] != A[j][i]:
                raise ValueError("Матрица должна быть симметрической.")
    if not isinstance(eps, (int, float)) or not isfinite(eps) or eps <= 0:
        raise ValueError("Точность должна быть конечной и положительной.")
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError(
            "Лимит вращений должен быть положительным целым числом.")


def off_diagonal_norm(A):
    """Условие малости t(A)"""
    total = 0.0
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            total += A[i][j] ** 2
    return sqrt(total)


def max_off_diagonal(A):
    """Выбор максимального элемента вне диагонали"""
    i_max, j_max = 0, 1
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if abs(A[i][j]) > abs(A[i_max][j_max]):
                i_max, j_max = i, j
    return i_max, j_max


def rotation_matrix(A, i, j):
    """Вычисление матрицы преобразования подобия U_step"""
    if A[i][i] == A[j][j]:
        phi = pi / 4
    else:
        phi = 0.5 * atan(2 * A[i][j] / (A[i][i] - A[j][j]))

    U_step = np.eye(len(A)).tolist()
    U_step[i][i] = cos(phi)
    U_step[j][j] = cos(phi)
    U_step[i][j] = -sin(phi)
    U_step[j][i] = sin(phi)
    return U_step


def calculate_residuals(A, B, U):
    """Вычисляет невязку для каждой собственной пары и наибольшую их длину.

    B[j][j] — приближённое собственное значение, столбец j матрицы U —
    соответствующий вектор. Их невязка A*v - lambda*v записывается
    в столбец j матрицы R. Длина невязки — корень из суммы квадратов её компонент.
    """
    R = np.matmul(A, U).tolist()
    max_norm = 0.0
    for j in range(len(A)):
        total = 0.0
        for i in range(len(A)):
            R[i][j] -= B[j][j] * U[i][j]
            total += R[i][j] ** 2
        max_norm = max(max_norm, sqrt(total))
    return R, max_norm


def rotation_method(A, eps, max_iter=10000):
    """Возвращает значения, матрицу векторов-столбцов U, число вращений,
    преобразованную матрицу и историю. Каждая запись истории содержит
    число вращений, критерий t и наибольшую длину невязки среди всех пар.
    """
    validate_inputs(A, eps, max_iter)
    n = len(A)
    B = []
    for row in A:
        B.append(row.copy())
    U = np.eye(n).tolist()
    history = []
    k = 0

    while True:
        t = off_diagonal_norm(B)
        _, max_residual = calculate_residuals(A, B, U)
        if not isfinite(t) or not isfinite(max_residual):
            raise RuntimeError("Переполнение: точность не достигнута.")
        history.append((k, t, max_residual))
        if t < eps:
            break
        if k == max_iter:
            raise RuntimeError(
                f"Точность {eps:g} не достигнута за {max_iter} вращений; "
                f"t = {t:.6e}."
            )

        i, j = max_off_diagonal(B)
        U_step = rotation_matrix(B, i, j)
        # U_step — матрица одного вращения U^(k): A^(k+1) = U^(k)^T * A^(k) * U^(k).
        B = np.matmul(np.matmul(np.transpose(U_step), B), U_step).tolist()
        # Собственные векторы — столбцы произведения U^(0)*U^(1)*... .
        U = np.matmul(U, U_step).tolist()
        k += 1

    values = []
    for j in range(n):
        values.append(B[j][j])
        # Теоретически столбцы уже единичные; устраняем ошибку округления длины.
        length = 0.0
        for i in range(n):
            length += U[i][j] ** 2
        length = sqrt(length)
        for i in range(n):
            U[i][j] /= length
    return values, U, k, B, history


def print_matrix(name, A):
    print(name)
    print(np.array2string(np.array(A), precision=4, suppress_small=True))


def print_result(A, eps, result):
    values, U, k, B, history = result
    n = len(A)
    print(f"\nТочность eps = {eps:g}; число вращений = {k}")
    print_matrix("Итоговая преобразованная матрица:", B)
    R, max_residual = calculate_residuals(A, B, U)
    for j in range(n):
        print(f"\nlambda_{j + 1} = {values[j]:.6f}")
        print("v =", end=" ")
        for i in range(n):
            print(f"{U[i][j]: .6f}", end=" ")
        print("\nA*v - lambda*v =", end=" ")
        for i in range(n):
            print(f"{R[i][j]: .2e}", end=" ")
        print()

    gram = np.matmul(np.transpose(U), U)
    print_matrix("\nU^T * U (должна быть близка к E):", gram)
    print(
        f"Максимальное отклонение от E: {np.max(np.abs(gram - np.eye(n))):.2e}")
    Lambda = []
    for i in range(n):
        row = [0.0] * n
        row[i] = values[i]
        Lambda.append(row)
    AU = np.matmul(A, U)
    UL = np.matmul(U, Lambda)
    print_matrix("A * U:", AU)
    print_matrix("U * Lambda:", UL)
    print(
        f"Максимальное отличие A*U и U*Lambda: {np.max(np.abs(AU - UL)):.2e}")
    print(f"Максимальная евклидова норма невязки: {max_residual:.2e}")

    print("\nИстория с начального состояния (k = 0):")
    print(f"{'k':>3} {'t(A^(k))':>14} {'max ||r_j||_2':>16}")
    for step, criterion, residual in history:
        print(f"{step:3d} {criterion:14.3g} {residual:16.3g}")


def main():
    A = [
        [-6, 6, -8],
        [6, -4, 9],
        [-8, 9, -2],
    ]
    eps = 1e-6
    print_matrix("Вариант №7. Исходная матрица:", A)
    print("Начальное состояние: A^(0) = A, накопленная U = E.")
    result = rotation_method(A, eps)
    print_result(A, eps, result)

    print("\nЗависимость результата от точности:")
    print(f"{'eps':>9} {'Вращений':>9} {'t':>12} {'max ||r||_2':>13}", end=" ")
    for j in range(len(A)):
        name = f"lambda_{j + 1}"
        print(f"{name:>12}", end=" ")
    print()
    for tolerance in [1e-2, 1e-4, 1e-6]:
        values, U, k, B, history = rotation_method(A, tolerance)
        _, max_residual = calculate_residuals(A, B, U)
        print(f"{tolerance:9g} {k:9d} {history[-1][1]:12.2e} {max_residual:13.2e}",
              end=" ")
        for value in values:
            print(f"{value:12.6f}", end=" ")
        print()
    print("t — величина элементов выше диагонали; невязка — нарушение A*v = lambda*v.")
    print("Фактическая погрешность — отличие от точной собственной пары.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OverflowError) as error:
        print(f"Ошибка: {error}")
        raise SystemExit(1)
