'''
1.3. Реализовать метод простых итераций и метод Зейделя в виде программ, 
задавая в качестве входных данных матрицу системы, вектор правых частей и точность вычислений. 
Используя разработанное программное обеспечение, решить СЛАУ. 
Проанализировать количество итераций, необходимое для достижения заданной точности.

№7
29 8 9 -9 | 197
-7 -25 0 9 | -226
1 6 16 -2 | -95
-7 4 -2 17 | -58
'''

from math import ceil, isfinite, log


def validate_inputs(A, b, eps, max_iter):
    n = len(A)
    if n == 0 or len(b) != n:
        raise ValueError(
            "Нужны непустая матрица A и вектор b той же размерности.")
    for row in A:
        if len(row) != n:
            raise ValueError("Матрица A должна быть квадратной.")
        for value in row:
            if not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError("Элементы A должны быть конечными числами.")
    for value in b:
        if not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("Элементы b должны быть конечными числами.")
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError(
                "Нулевой диагональный элемент: требуется перестановка строк.")
    if not isinstance(eps, (int, float)) or not isfinite(eps) or eps <= 0:
        raise ValueError("Точность eps должна быть конечной и положительной.")
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError(
            "Лимит итераций должен быть положительным целым числом.")


def vector_norm(vector):
    """Норма ||x||_c — наибольший модуль компоненты."""
    return max(abs(value) for value in vector)


def matrix_norm(matrix):
    """Согласованная норма ||A||_c — максимум сумм модулей по строкам."""
    row_sums = []
    for row in matrix:
        row_sums.append(sum(abs(value) for value in row))
    return max(row_sums)


def prepare_system(A, b):
    """Формула (1.18): beta_i = b_i/a_ii, alpha_ij = -a_ij/a_ii."""
    n = len(A)
    alpha = []
    beta = []
    for i in range(n):
        row = []
        beta.append(b[i] / A[i][i])
        for j in range(n):
            if i == j:
                row.append(0.0)
            else:
                row.append(-A[i][j] / A[i][i])
        if not isfinite(beta[i]) or not all(isfinite(value) for value in row):
            raise ValueError(
                "Переполнение при делении на диагональный элемент; измените масштаб системы.")
        alpha.append(row)
    return alpha, beta


def upper_part(alpha):
    """C из разложения alpha = B + C: диагональ и элементы выше неё."""
    n = len(alpha)
    C = []
    for i in range(n):
        row = [0.0] * n
        for j in range(i, n):
            row[j] = alpha[i][j]
        C.append(row)
    return C


def convergence_norm(alpha):
    q = matrix_norm(alpha)
    # При q >= 1 эти оценки погрешности не гарантируют точность.
    if not isfinite(q) or q >= 1:
        raise ValueError(
            "Условие ||alpha||_c < 1 не выполнено; оценка точности неприменима.")
    return q


def simple_iter(A, b, eps, max_iter=10000):
    validate_inputs(A, b, eps, max_iter)
    alpha, beta = prepare_system(A, b)
    q = convergence_norm(alpha)
    n = len(A)
    x = beta.copy()  # Одинаковое начальное приближение для обоих методов.
    if q == 0 or vector_norm(beta) == 0:
        return x, 0, 0.0

    for k in range(1, max_iter + 1):
        x_prev = x.copy()
        # Формула (1.19): все компоненты берутся с предыдущей итерации.
        for i in range(n):
            x[i] = beta[i]
            for j in range(n):
                x[i] += alpha[i][j] * x_prev[j]

        delta = vector_norm([x[i] - x_prev[i] for i in range(n)])
        error = q / (1 - q) * delta  # Оценка (1.20).
        if not all(isfinite(value) for value in x) or not isfinite(error):
            raise RuntimeError(
                "Простые итерации: переполнение, точность не достигнута.")
        if error <= eps:
            return x, k, error

    raise RuntimeError(
        f"Простые итерации: точность {eps:g} не достигнута за {max_iter} итераций; "
        f"последняя оценка погрешности {error:.6e}."
    )


def seidel(A, b, eps, max_iter=10000):
    validate_inputs(A, b, eps, max_iter)
    alpha, beta = prepare_system(A, b)
    q = convergence_norm(alpha)
    norm_C = matrix_norm(upper_part(alpha))
    n = len(A)
    x = beta.copy()
    if q == 0 or vector_norm(beta) == 0:
        return x, 0, 0.0

    for k in range(1, max_iter + 1):
        x_prev = x.copy()
        for i in range(n):
            x[i] = beta[i]
            # B*x^(k): уже вычисленные компоненты текущей итерации.
            for j in range(i):
                x[i] += alpha[i][j] * x[j]
            # C*x^(k-1): оставшиеся компоненты предыдущей итерации.
            for j in range(i, n):
                x[i] += alpha[i][j] * x_prev[j]

        delta = vector_norm([x[i] - x_prev[i] for i in range(n)])
        # e^(k) = alpha*e^(k) + C*(x^(k-1) - x^(k)), e^(k) = x^(k) - x*.
        # (1 - ||alpha||_c)*||e^(k)||_c <= ||C||_c*delta, методичка, с. 19.
        error = norm_C / (1 - q) * delta
        if not all(isfinite(value) for value in x) or not isfinite(error):
            raise RuntimeError(
                "Зейдель: переполнение, точность не достигнута.")
        if error <= eps:
            return x, k, error

    raise RuntimeError(
        f"Зейдель: точность {eps:g} не достигнута за {max_iter} итераций; "
        f"последняя оценка погрешности {error:.6e}."
    )


def apriori_count(q, beta, eps):
    """Достаточное число итераций Якоби при x^(0) = beta (методичка, с. 17)."""
    norm_beta = vector_norm(beta)
    if q == 0 or norm_beta == 0:
        return 0
    # q^(k+1)*||beta||_c/(1-q) <= eps. Правая часть даёт границу для k+1.
    bound = (log(eps) - log(norm_beta) + log(1 - q)) / log(q)
    return max(0, ceil(bound) - 1)


def residual(A, b, x):
    """Невязка r = A*x - b в исходной системе."""
    r = []
    for i in range(len(A)):
        value = -b[i]
        for j in range(len(A)):
            value += A[i][j] * x[j]
        r.append(value)
    return r


def print_vector(name, vector):
    print(name, "[" + ", ".join(f"{value:.10f}" for value in vector) + "]")


def print_result(name, A, b, result):
    x, _, error = result
    r = residual(A, b, x)
    print_vector(f"\n{name}: x =", x)
    print(f"Проверка: оценка погрешности = {error:.3e}; "
          f"||A*x - b||_c = {vector_norm(r):.3e}")


def main():
    A = [
        [29, 8, 9, -9],
        [-7, -25, 0, 9],
        [1, 6, 16, -2],
        [-7, 4, -2, 17],
    ]
    b = [197, -226, -95, -58]
    eps = 1e-3  # Основная точность вычислений.
    max_iter = 10000
    validate_inputs(A, b, eps, max_iter)

    print("Вариант №7. Исходные данные A | b:")
    for i in range(len(A)):
        print(" ".join(f"{value:4g}" for value in A[i]), f"| {b[i]:4g}")
    print(f"Точность eps = {eps:g}")

    jacobi = simple_iter(A, b, eps, max_iter)
    gauss_seidel = seidel(A, b, eps, max_iter)
    print_result("Простые итерации", A, b, jacobi)
    print_result("Зейдель", A, b, gauss_seidel)

    print("\nЧисло итераций:")
    print(f"{'eps':>8} {'Простые итерации':>18} {'Зейдель':>9}")
    for tolerance in [1e-2, eps, 1e-4]:
        if tolerance == eps:
            jacobi_count, seidel_count = jacobi[1], gauss_seidel[1]
        else:
            jacobi_count = simple_iter(A, b, tolerance, max_iter)[1]
            seidel_count = seidel(A, b, tolerance, max_iter)[1]
        print(f"{tolerance:8g} {jacobi_count:18} {seidel_count:9}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        print(f"Ошибка: {error}")
        raise SystemExit(1)
