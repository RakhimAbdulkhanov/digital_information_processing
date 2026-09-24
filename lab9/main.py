import time
import numpy as np
import matplotlib.pyplot as plt


def discrete_linear_convolution(x, y):
    # пряма лінійна дискретна згортка за формулою f[m] = sum(x[k] * y[m - k])
    n = len(x)
    m = len(y)
    out_len = n + m - 1
    result = np.zeros(out_len)

    for i in range(out_len):
        step_sum = 0.0
        for k in range(n):
            y_index = i - k
            if 0 <= y_index < m:
                step_sum += x[k] * y[y_index]
        result[i] = step_sum

    return result


def fast_convolution_fft(x, y):
    # швидка згортка на основі теореми про згортку та шпф
    n = len(x)
    m = len(y)
    out_len = n + m - 1

    # доповнення нулями до повної довжини лінійної згортки
    x_fft = np.fft.fft(x, out_len)
    y_fft = np.fft.fft(y, out_len)

    # множення спектрів та обернене шпф
    result = np.real(np.fft.ifft(x_fft * y_fft))
    return result


def circular_convolution(x, y, N=None):
    # кругова циклічна згортка довжини n
    if N is None:
        N = max(len(x), len(y))

    x_pad = np.pad(x, (0, max(0, N - len(x))))[:N]
    y_pad = np.pad(y, (0, max(0, N - len(y))))[:N]
    result = np.zeros(N)

    for n_idx in range(N):
        step_sum = 0.0
        for k in range(N):
            step_sum += x_pad[k] * y_pad[(n_idx - k) % N]
        result[n_idx] = step_sum

    return result


def plot_convolution_analysis(x, y, linear_res, fft_res, circ_unpadded, circ_padded, title):
    # візуалізація вхідних сигналів, лінійної та кругової згорток
    fig, axes = plt.subplots(4, 1, figsize=(11, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # вхідний сигнал x[k]
    k_x = np.arange(len(x))
    axes[0].stem(k_x, x, linefmt="b-", markerfmt="bo", basefmt="k-")
    axes[0].set_title("Вхідний сигнал x[k]", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("x[k]")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # вхідний сигнал y[k] (імпульсна характеристика)
    k_y = np.arange(len(y))
    axes[1].stem(k_y, y, linefmt="g-", markerfmt="go", basefmt="k-")
    axes[1].set_title("Вхідний сигнал y[k] (імпульсна характеристика)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("y[k]")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # результат лінійної згортки f[m]
    m_lin = np.arange(len(linear_res))
    axes[2].stem(m_lin, linear_res, linefmt="r-", markerfmt="ro", basefmt="k-", label="пряма згортка")
    axes[2].plot(m_lin, fft_res, "k--", alpha=0.7, label="швидка згортка (шпф)")
    axes[2].set_title(f"Лінійна згортка f[m] = x[k] * y[k] (довжина N={len(linear_res)})", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("m")
    axes[2].set_ylabel("f[m]")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right")

    # порівняння кругової згортки з доповненням і без
    m_circ_unpad = np.arange(len(circ_unpadded))
    m_circ_pad = np.arange(len(circ_padded))
    axes[3].stem(m_circ_pad, circ_padded, linefmt="c-", markerfmt="co", basefmt="k-", label=f"кругова з padding (N={len(circ_padded)})")
    axes[3].stem(m_circ_unpad, circ_unpadded, linefmt="m--", markerfmt="ms", basefmt="k-", label=f"кругова без padding (N={len(circ_unpadded)})")
    axes[3].set_title("Кругова згортка: з доповненням нулями (еквівалент лінійної) vs без доповнення", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("m")
    axes[3].set_ylabel("y_circ[m]")
    axes[3].grid(True, linestyle="--", alpha=0.5)
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def benchmark_convolutions():
    # порівняння швидкодії прямої та швидкої згортки на довгих сигналах
    lengths = [64, 128, 256, 512, 1024, 2048]
    time_direct = []
    time_fft = []

    for length in lengths:
        sig_a = np.random.rand(length)
        sig_b = np.random.rand(length)

        # час прямої згортки
        t0 = time.perf_counter()
        _ = discrete_linear_convolution(sig_a, sig_b)
        t_dir = (time.perf_counter() - t0) * 1000.0
        time_direct.append(t_dir)

        # час швидкої згортки
        t0 = time.perf_counter()
        _ = fast_convolution_fft(sig_a, sig_b)
        t_f = (time.perf_counter() - t0) * 1000.0
        time_fft.append(t_f)

    return lengths, time_direct, time_fft


def main():
    # вхідні послідовності з таблиці 1
    datasets = [
        {
            "name": "Варіант 1",
            "x": np.array([3, 5, 7, 4, 5, 7, 3, 6, 4, 7, 8, 1, 1, 5], dtype=float),
            "y": np.array([6, 3, 3, 4, 1, 6, 7, 5, 3, 6, 7, 1, 0], dtype=float)
        },
        {
            "name": "Варіант 3",
            "x": np.array([0, 8, 6, 3, 2, 4, 0, 1, 2, 8, 7, 3, 4, 6, 0, 9, 2, 1, 6, 4, 0], dtype=float),
            "y": np.array([4, 2, 2, 1, 8, 7, 6, 4, 0, 1, 2, 3, 6, 4], dtype=float)
        }
    ]

    for data in datasets:
        name = data["name"]
        x = data["x"]
        y = data["y"]

        nx = len(x)
        ny = len(y)
        expected_len = nx + ny - 1

        # розрахунок прямої згортки
        res_direct = discrete_linear_convolution(x, y)

        # розрахунок швидкої згортки через шпф
        res_fft = fast_convolution_fft(x, y)

        # перевірка через numpy
        res_np = np.convolve(x, y)

        # кругова згортка без доповнення та з доповненням нулями
        res_circ_unpad = circular_convolution(x, y, max(nx, ny))
        res_circ_pad = circular_convolution(x, y, expected_len)

        print(f"=== {name} ===")
        print(f"Довжина x[k]: {nx}, довжина y[k]: {ny}")
        print(f"Розрахункова довжина згортки: {expected_len}")
        print(f"Перші 5 відліків: {res_direct[:5]}")
        print(f"Останні 5 відліків: {res_direct[-5:]}")
        print(f"Максимальне значення згортки: {np.max(res_direct):.2f}")

        # звірка результатів
        match_direct_np = np.allclose(res_direct, res_np)
        match_fft_np = np.allclose(res_fft, res_np)
        match_circ_pad = np.allclose(res_circ_pad, res_direct)

        print(f"Пряма згортка == np.convolve: {match_direct_np}")
        print(f"Швидка згортка (ШПФ) == np.convolve: {match_fft_np}")
        print(f"Кругова згортка з padding == Лінійна згортка: {match_circ_pad}\n")

        # відображення графіків
        plot_convolution_analysis(
            x, y, res_direct, res_fft, res_circ_unpad, res_circ_pad,
            f"Дискретна згортка сигналів - {name}"
        )

    # порівняння часу виконання
    lengths, t_dir, t_fft = benchmark_convolutions()
    print("=== Порівняння швидкодії (мс) ===")
    print(f"{'Довжина N':<12} | {'Пряма O(N^2)':<16} | {'Швидка ШПФ O(N log N)':<22} | {'Прискорення':<12}")
    print("-" * 70)
    for l_val, td, tf in zip(lengths, t_dir, t_fft):
        speedup = td / tf if tf > 0 else 0
        print(f"{l_val:<12} | {td:<16.4f} | {tf:<22.4f} | {speedup:<12.2f}x")

    # графік швидкодії
    plt.figure(figsize=(9, 5))
    plt.plot(lengths, t_dir, "o-", color="#d62728", label="Пряма лінійна згортка O(N^2)")
    plt.plot(lengths, t_fft, "s-", color="#1f77b4", label="Швидка згортка через ШПФ O(N log N)")
    plt.title("Порівняння обчислювальної складності дискретної згортки", fontsize=12, fontweight="bold")
    plt.xlabel("Довжина вхідних послідовностей N")
    plt.ylabel("Час обчислення (мс)")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
