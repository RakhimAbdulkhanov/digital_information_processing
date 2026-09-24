import numpy as np
import matplotlib.pyplot as plt
import timeit
import warnings

# глушимо ділення на нуль при обчисленні періоду T = 1 / nu
warnings.filterwarnings("ignore", category=RuntimeWarning)

# поворотний множник
def W(k, N):
    return np.exp(-2j * np.pi * k / N)

# класичне дпф зі складністю O(N^2)
def dft(x):
    N = len(x)
    return [sum(x[n] * np.exp(-2j * np.pi * k * n / N) for n in range(N)) for k in range(N)]

# двоетапне розбиття дпф
def dft2(x):
    N = len(x)
    if N % 2 != 0:
        raise ValueError("кількість відліків має бути парною")
    x1 = x[::2]
    x2 = x[1::2]
    X1 = dft(x1)
    X2 = dft(x2)
    return [X1[k] + W(k, N) * X2[k] for k in range(N // 2)] + [X1[k] - W(k, N) * X2[k] for k in range(N // 2)]

# швидке перетворення фур'є кулі-тюкі за основою 2
def fft(x):
    N = len(x)
    if N & (N - 1) != 0:
        raise ValueError("кількість відліків має бути степенем двійки")
    if N <= 1:
        return x
    if N == 2:
        return [x[0] + x[1], x[0] - x[1]]

    # рекурсивно ділимо на парні та непарні індекси
    x1 = fft(x[0::2])
    x2 = fft(x[1::2])

    Y1 = []
    Y2 = []
    for k in range(N // 2):
        tmp = W(k, N) * x2[k]
        Y1.append(x1[k] + tmp)
        Y2.append(x1[k] - tmp)
    return Y1 + Y2

# обернене шпф через пряме
def ifft(X):
    N = len(X)
    X_conj = [np.conj(val) for val in X]
    x_trans = fft(X_conj)
    return [np.conj(val) / N for val in x_trans]

# -------------------------------------------------------------
# 1. дослідження поворотного множника W(k, N)
# -------------------------------------------------------------
k_arr = np.arange(100)
w_arr = [W(k, 20) for k in k_arr]

plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(k_arr, np.real(w_arr), color="#1f77b4", linewidth=1.5)
plt.title("Дійсна частина Re(W(k, 20))")
plt.xlabel("k")
plt.ylabel("Re(W)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(k_arr, np.imag(w_arr), color="#d62728", linewidth=1.5)
plt.title("Уявна частина Im(W(k, 20))")
plt.xlabel("k")
plt.ylabel("Im(W)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 2. генерація тестового сигналу за формулою (12)
# -------------------------------------------------------------
dt = 0.7
N = 512
t = np.arange(N) * dt
x = np.cos(2 * np.pi * t / 10 + 1) + np.cos(2 * np.pi * t / 40 + np.pi / 2)

plt.figure(figsize=(12, 4))
plt.plot(t, x, color="#2ca02c", linewidth=1.3)
plt.title("Модельний сигнал X(t) = cos(2*pi*t/10 + 1) + cos(2*pi*t/40 + pi/2)")
plt.xlabel("Час t (с)")
plt.ylabel("X(t)")
plt.xlim(0, max(t))
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 3. вимірювання часу виконання dft vs fft
# -------------------------------------------------------------
xp2 = list(x[:N])

t_dft = timeit.timeit(lambda: dft(xp2), number=1)
t_fft = timeit.timeit(lambda: fft(xp2), number=1)
speedup = t_dft / t_fft

print(f"DFT time (N={N}): {t_dft:.6f} s")
print(f"FFT time (N={N}): {t_fft:.6f} s")
print(f"Speedup: {speedup:.2f}x")

# -------------------------------------------------------------
# 4. спектральний аналіз та періодограми
# -------------------------------------------------------------
X_dft = np.array(dft(xp2))
X_fft = np.array(fft(xp2))

A_dft = np.abs(X_dft)
P_dft = np.angle(X_dft)

A_fft = np.abs(X_fft)
P_fft = np.angle(X_fft)

nu = np.arange(N) / dt / N
T = 1.0 / nu

# періодограма через дпф
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.semilogy(T[0:N//2], A_dft[0:N//2], color="#1f77b4", linewidth=1.5)
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, step=10))
plt.title("Амплітудна періодограма (ДПФ)")
plt.xlabel("Період T (с)")
plt.ylabel("Амплітуда |X| (лог. шкала)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(T[0:N//2], P_dft[0:N//2], color="#d62728", linewidth=1.2)
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, step=10))
plt.title("Фазова періодограма (ДПФ)")
plt.xlabel("Період T (с)")
plt.ylabel("Фаза (рад)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# періодограма через шпф
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.semilogy(T[0:N//2], A_fft[0:N//2], color="#2ca02c", linewidth=1.5)
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, step=10))
plt.title("Амплітудна періодограма (ШПФ)")
plt.xlabel("Період T (с)")
plt.ylabel("Амплітуда |X| (лог. шкала)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(T[0:N//2], P_fft[0:N//2], color="#9467bd", linewidth=1.2)
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, step=10))
plt.title("Фазова періодограма (ШПФ)")
plt.xlabel("Період T (с)")
plt.ylabel("Фаза (рад)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# порівняльна гістограма швидкодії
methods = ["ДПФ (O(N^2))", "ШПФ Кулі-Тюкі (O(N log2 N))"]
times = [t_dft, t_fft]
colors = ["#d62728", "#2ca02c"]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(methods, times, color=colors, width=0.45)
plt.ylabel("Час виконання (с)")
plt.title(f"Порівняння швидкодії алгоритмів перетворення Фур'є (N = {N})")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(times), f"{yval:.6f} с", ha="center", va="bottom", fontweight="bold")
plt.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
