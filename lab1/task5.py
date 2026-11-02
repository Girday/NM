'''
1.5. Реализовать алгоритм QR – разложения матриц в виде программы. 
На его основе разработать программу, реализующую QR – алгоритм решения полной проблемы собственных 
значений произвольных матриц, задавая в качестве входных данных матрицу и точность вычислений. 
С использованием разработанного программного обеспечения найти собственные значения матрицы.

№7
9 0 2
-6 4 4
-2 -7 5
'''

from cmath import sqrt as complex_sqrt
from math import isfinite, sqrt

import numpy as np


def validate_matrix(A):
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


def qr_decomposition(A):
    """Отражения Хаусхолдера: формулы (1.29), (1.30), методичка, с. 28–29."""
    validate_matrix(A)
    n = len(A)
    R = [row.copy() for row in A]
    Q = np.eye(n).tolist()

    for j in range(n - 1):
        # Берём часть j-го столбца, начиная с диагонали.
        scale = max(abs(R[i][j]) for i in range(j, n))
        if scale == 0:
            continue

        v = [0.0] * n
        for i in range(j, n):
            v[i] = R[i][j] / scale

        length = sqrt(sum(v[i] ** 2 for i in range(j, n)))
        sign = 1 if v[j] >= 0 else -1
        v[j] += sign * length

        denominator = sum(value ** 2 for value in v)
        if denominator == 0 or not isfinite(denominator):
            raise RuntimeError(
                "Не удалось построить отражение: недопустимое v^T*v.")

        H = np.eye(n).tolist()
        for i in range(j, n):
            for k in range(j, n):
                H[i][k] -= 2 * v[i] * v[k] / denominator

        # R = H_(n-1)*...*H_1*A, Q = H_1*...*H_(n-1).
        R = np.matmul(H, R).tolist()
        Q = np.matmul(Q, H).tolist()
        if not np.isfinite(R).all() or not np.isfinite(Q).all():
            raise RuntimeError("Переполнение при QR-разложении.")

    return Q, R


def column_tail_norm(A, column, start):
    """Длина части столбца ниже строки start - 1."""
    total = 0.0
    for i in range(start, len(A)):
        total += A[i][column] ** 2

    return sqrt(total)


def block_eigenvalues(A, j):
    """Корни (a - lambda)*(d - lambda) = b*c для блока 2x2, с. 31."""
    a, b = A[j][j], A[j][j + 1]
    c, d = A[j + 1][j], A[j + 1][j + 1]
    discriminant = (a - d) ** 2 + 4 * b * c

    if not isfinite(discriminant):
        raise RuntimeError("Переполнение при вычислении корней блока 2x2.")

    root = complex_sqrt(discriminant)

    return [(a + d + root) / 2, (a + d - root) / 2]


def converged_values(A, previous, eps):
    """Критерии с. 30–31: отделение блоков и устойчивость их значений."""
    values = []
    n = len(A)
    j = 0

    while j < n:
        if column_tail_norm(A, j, j + 1) <= eps:
            # Вещественное значение: малость столбца под диагональю.
            if abs(A[j][j] - previous[j][j]) > eps:
                return None
            values.append(A[j][j])
            j += 1
        else:
            if j + 1 == n:
                return None
            # Под блоком должны быть малы ОБА столбца; внутри блока нуль не нужен.
            left = column_tail_norm(A, j, j + 2)
            right = column_tail_norm(A, j + 1, j + 2)
            if sqrt(left ** 2 + right ** 2) > eps:
                return None
            pair = block_eigenvalues(A, j)
            old_pair = block_eigenvalues(previous, j)
            if pair[0].imag == 0 or old_pair[0].imag == 0:
                return None
            # Сопоставляем корни, учитывая возможную перестановку пары.
            same = max(abs(pair[0] - old_pair[0]), abs(pair[1] - old_pair[1]))
            swapped = max(abs(pair[0] - old_pair[1]),
                          abs(pair[1] - old_pair[0]))
            if min(same, swapped) > eps:
                return None
            values.extend(pair)
            j += 2

    return values


def qr_algorithm(A, eps, max_iter=10000):
    """A^(k) = Q^(k)*R^(k), A^(k+1) = R^(k)*Q^(k), методичка, с. 30."""
    validate_matrix(A)
    if not isinstance(eps, (int, float)) or not isfinite(eps) or eps <= 0:
        raise ValueError("Точность должна быть конечной и положительной.")
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError(
            "Лимит итераций должен быть положительным целым числом.")
    B = [row.copy() for row in A]

    for k in range(1, max_iter + 1):
        previous = B
        Q, R = qr_decomposition(previous)
        B = np.matmul(R, Q).tolist()
        if not np.isfinite(B).all():
            raise RuntimeError(
                "QR-алгоритм: переполнение, точность не достигнута.")
        values = converged_values(B, previous, eps)
        if values is not None:
            return values, k, B

    raise RuntimeError(
        f"QR-алгоритм: точность {eps:g} не достигнута за {max_iter} итераций."
    )


def print_matrix(name, A):
    print(name)
    print(np.array2string(np.array(A), precision=4, suppress_small=True))
    print()


def print_qr_check(A, Q, R):
    reconstruction = np.max(np.abs(np.matmul(Q, R) - np.array(A)))
    orthogonality = np.max(
        np.abs(np.matmul(np.transpose(Q), Q) - np.eye(len(A))))
    below_diagonal = 0.0
    for i in range(len(A)):
        for j in range(i):
            below_diagonal = max(below_diagonal, abs(R[i][j]))
    print("Проверка QR-разложения (максимальные модули элементов):")
    print(f"Q*R - A:   {reconstruction:.2e}")
    print(f"Q^T*Q - E: {orthogonality:.2e}")
    print(f"R ниже главной диагонали: {below_diagonal:.2e}")


def format_value(value):
    if isinstance(value, complex) and value.imag != 0:
        return f"{value.real:.6f} {value.imag:+.6f}i"

    return f"{value.real:.6f}".rstrip("0").rstrip(".")


def print_independent_check(A, values):
    """Библиотечный решатель применяется только для независимой проверки."""
    reference = list(np.linalg.eigvals(np.array(A, dtype=float)))
    max_error = 0.0
    print("\nПроверка NumPy (значения сопоставлены по близости):")
    for value in values:
        closest = 0
        for i in range(1, len(reference)):
            if abs(value - reference[i]) < abs(value - reference[closest]):
                closest = i
        expected = reference.pop(closest)
        error = abs(value - expected)
        max_error = max(max_error, error)
        print(f"  {format_value(complex(expected))}; отличие {error:.2e}")
    trace = sum(A[i][i] for i in range(len(A)))
    total = sum(values)
    print(f"\nНаибольшее отличие: {max_error:.2e}")
    print(f"\nСумма lambda = {format_value(complex(total))}")
    print(f"След A = {trace:g}")
    print(f"|Сумма lambda - след A| = {abs(total - trace):.2e}")


def main():
    A = [
        [9, 0, 2],
        [-6, 4, 4],
        [-2, -7, 5],
    ]
    eps = 1e-6
    print_matrix("Вариант №7. Исходная матрица A:", A)
    Q, R = qr_decomposition(A)
    print_matrix("Q:", Q)
    print_matrix("R:", R)
    print_qr_check(A, Q, R)

    print(f"\nОсновная точность eps = {eps:g}")
    print("Начальное приближение A^(0) = A.")
    print("Критерии: отделение блоков и изменение значений не больше eps.")
    for tolerance in [1e-3, eps]:
        values, k, B = qr_algorithm(A, tolerance)
        print(f"\n{'=' * 48}")
        print(f"eps = {tolerance:g}; число QR-итераций = {k}\n")
        print_matrix(
            "Итоговая матрица QR-процесса (округлена до 4 знаков):", B)
        print("Собственные значения:")
        for j, value in enumerate(values, start=1):
            print(f"lambda_{j} = {format_value(value)}")
        print_independent_check(A, values)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OverflowError) as error:
        print(f"Ошибка: {error}")
        raise SystemExit(1)
