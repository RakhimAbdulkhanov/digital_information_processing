import numpy as np
import matplotlib.pyplot as plt

# параметри сигналу
n_points = 1000
n = np.arange(n_points)

# модельний сигнал за варіантом
x = 3.0 * np.sin(n / 5.0) + 3.0 * np.sin(n / 6.0)

# ручна матрична реалізація прямого дпф
def dft_manual(signal):
    n_pts = len(signal)
    n_idx = np.arange(n_pts)
    k_idx = n_idx.reshape((n_pts, 1))
    w_mat = np.exp(-2j * np.pi * k_idx * n_idx / n_pts)
    return np.dot(w_mat, signal)

# ручна матрична реалізація зворотного дпф
def idft_manual(spectrum):
    n_pts = len(spectrum)
    k_idx = np.arange(n_pts)
    n_idx = k_idx.reshape((n_pts, 1))
    w_inv = np.exp(2j * np.pi * n_idx * k_idx / n_pts)
    return (1.0 / n_pts) * np.dot(w_inv, spectrum)

# розрахунок прямих спектрів
x_dft = dft_manual(x)
x_fft = np.fft.fft(x)

# амплітудний і фазовий спектри
freq = np.arange(n_points)
amplitude = np.abs(x_fft)
phase = np.angle(x_fft)

# відновлення сигналу оберненим перетворенням
x_rec_manual = idft_manual(x_dft).real
x_rec_fft = np.fft.ifft(x_fft).real

# розрахунок числових похибок і перевірка симетрії
dft_diff = np.max(np.abs(x_dft - x_fft))
rec_err_manual = np.max(np.abs(x - x_rec_manual))
rec_err_fft = np.max(np.abs(x - x_rec_fft))
mse_manual = np.mean((x - x_rec_manual) ** 2)
mse_fft = np.mean((x - x_rec_fft) ** 2)
half_n = n_points // 2
sym_diff = np.max(np.abs(x_fft[1:half_n] - np.conj(x_fft[n_points - 1:half_n:-1])))

print(f"Параметри сигналу: N = {n_points} відліків")
print("Модельний ряд: x[n] = 3*sin(n/5) + 3*sin(n/6)")
print("\nАналіз основних гармонік у додатній половині частот:")
print(f"1. Складова w1 = 1/6 (~0.1667 рад/відлік): пік k = 27, |X[k]| = {amplitude[27]:.2f}, фаза = {phase[27]:.4f} рад")
print(f"2. Складова w2 = 1/5 (~0.2000 рад/відлік): пік k = 32, |X[k]| = {amplitude[32]:.2f}, фаза = {phase[32]:.4f} рад")
print("\nДзеркальні спряжені піки у другій половині частот:")
print(f"1. k = {n_points - 32} (N - 32): |X[k]| = {amplitude[n_points - 32]:.2f}, фаза = {phase[n_points - 32]:.4f} рад")
print(f"2. k = {n_points - 27} (N - 27): |X[k]| = {amplitude[n_points - 27]:.2f}, фаза = {phase[n_points - 27]:.4f} рад")
print("\nОцінка точності та похибок:")
print(f"Максимальна різниця прямого ДПФ (матричне vs FFT): {dft_diff:.3e}")
print(f"Максимальне відхилення комплексної спряженості:   {sym_diff:.3e}")
print(f"Похибка відновлення ручним IDFT: Max = {rec_err_manual:.3e}, MSE = {mse_manual:.3e}")
print(f"Похибка відновлення бібліотечним IFFT: Max = {rec_err_fft:.3e}, MSE = {mse_fft:.3e}")

# побудова графіків за завданням лабораторної роботи
plt.figure(figsize=(11, 9))

# рис.1. початковий сигнал
plt.subplot(4, 1, 1)
plt.plot(n, x, color="#1f77b4", linewidth=1.2)
plt.title("Рис.1. Графічний вираз функції x[n] = 3*sin(n/5) + 3*sin(n/6)")
plt.xlabel("n")
plt.ylabel("Амплітуда")
plt.xlim(0, n_points)
plt.grid(True, linestyle="--", alpha=0.6)

# рис.2. фаза дпф
plt.subplot(4, 1, 2)
plt.plot(freq, phase, color="#ff7f0e", linewidth=1.0)
plt.title("Рис.2. Графік фази ДПФ arg(X[k])")
plt.xlabel("k")
plt.ylabel("Фаза, рад")
plt.xlim(0, n_points)
plt.grid(True, linestyle="--", alpha=0.6)

# рис.3. амплітуда дпф у логарифмічному масштабі
plt.subplot(4, 1, 3)
plt.semilogy(freq, amplitude, color="#2ca02c", linewidth=1.2)
plt.title("Рис.3. Графік амплітуди ДПФ |X[k]| (логарифмічна шкала)")
plt.xlabel("k")
plt.ylabel("|X[k]|")
plt.xlim(0, n_points)
plt.grid(True, which="both", linestyle="--", alpha=0.6)

# рис.4. зворотне перетворення
plt.subplot(4, 1, 4)
plt.plot(n, x_rec_manual, color="#d62728", linewidth=1.2)
plt.title("Рис.4. Зворотне перетворення x_rec[n]")
plt.xlabel("n")
plt.ylabel("Амплітуда")
plt.xlim(0, n_points)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
