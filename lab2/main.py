import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. неперервні (аналогові) сигнали
# -------------------------------------------------------------

# базова функція неперервного сигналу (експоненційно затухаючий синус)
def analog_signal(t):
    return np.where(t >= 0, np.exp(-0.4 * t) * np.sin(2 * np.pi * 0.8 * t), 0.0)

# часова сітка для неперервних сигналів
t = np.linspace(-4.0, 10.0, 1400)
x_t = analog_signal(t)

# масштабування амплітуди y(t) = a * x(t)
scale_a = 2.0
y_scaled = scale_a * x_t

# реверс за часом y(t) = x(-t)
y_reversed = analog_signal(-t)

# зсув у часі: затримка (tau = 2) та випередження (tau = -2)
tau_delay = 2.5
tau_adv = -2.0
y_delayed = analog_signal(t - tau_delay)
y_advanced = analog_signal(t - tau_adv)

# масштабування за часом: стиснення (a = 2) та розширення (a = 0.5)
y_compressed = analog_signal(2.0 * t)
y_expanded = analog_signal(0.5 * t)

# другий сигнал для накладання (опорний косинус)
x2_t = 0.6 * np.cos(2 * np.pi * 0.3 * t)

# додавання сигналів y(t) = x1(t) + x2(t)
y_added = x_t + x2_t

# множення сигналів (модуляція) y(t) = x1(t) * x2(t)
y_multiplied = x_t * x2_t

print("=== НЕПЕРЕРВНІ СИГНАЛИ ===")
print(f"Діапазон часу: [{t[0]:.1f}, {t[-1]:.1f}] с, кількість точок: {len(t)}")
print(f"Максимум базового сигналу: {np.max(x_t):.4f}")
print(f"Максимум масштабованого сигналу (a={scale_a}): {np.max(y_scaled):.4f}")
print(f"Енергія базового сигналу: {np.trapezoid(x_t**2, t):.4f}")
print(f"Енергія після додавання: {np.trapezoid(y_added**2, t):.4f}")

# графік 1: базовий сигнал, масштабування та реверс
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.plot(t, x_t, label="x(t) базовий", color="#1f77b4", linewidth=2)
plt.plot(t, y_scaled, label=f"y(t) = {scale_a}*x(t)", color="#d62728", linestyle="--", linewidth=1.8)
plt.title("Масштабування амплітуди неперервного сигналу")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, x_t, label="x(t)", color="#1f77b4", linewidth=2)
plt.plot(t, y_reversed, label="y(t) = x(-t)", color="#ff7f0e", linewidth=1.8)
plt.title("Реверс неперервного сигналу за часом")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# графік 2: зсув у часі та часове масштабування
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.plot(t, x_t, label="x(t)", color="#1f77b4", linewidth=2)
plt.plot(t, y_delayed, label=f"Затримка: x(t - {tau_delay})", color="#2ca02c", linestyle="--", linewidth=1.8)
plt.plot(t, y_advanced, label=f"Випередження: x(t + {abs(tau_adv)})", color="#9467bd", linestyle=":", linewidth=1.8)
plt.title("Зсув неперервного сигналу у часі")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, x_t, label="x(t)", color="#1f77b4", linewidth=2)
plt.plot(t, y_compressed, label="Стиснення: x(2t)", color="#8c564b", linewidth=1.8)
plt.plot(t, y_expanded, label="Розширення: x(0.5t)", color="#e377c2", linestyle="--", linewidth=1.8)
plt.title("Масштабування осі часу неперервного сигналу")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# графік 3: накладання неперервних сигналів (додавання та множення)
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.plot(t, x_t, label="x1(t)", color="#1f77b4", alpha=0.7)
plt.plot(t, x2_t, label="x2(t) опорний", color="#7f7f7f", linestyle=":", alpha=0.7)
plt.plot(t, y_added, label="x1(t) + x2(t)", color="#d62728", linewidth=2)
plt.title("Додавання неперервних сигналів")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, x_t, label="x1(t) обвідна", color="#1f77b4", linestyle="--", alpha=0.7)
plt.plot(t, y_multiplied, label="x1(t) * x2(t)", color="#2ca02c", linewidth=2)
plt.title("Множення (модуляція) неперервних сигналів")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 2. дискретні сигнали
# -------------------------------------------------------------

# генерація базової дискретної послідовності x[n]
n = np.arange(-6, 16)

def discrete_base(n_arr):
    # імпульсна затухаюча послідовність
    vals = np.zeros_like(n_arr, dtype=float)
    mask = n_arr >= 0
    vals[mask] = (0.82 ** n_arr[mask]) * np.sin(0.35 * np.pi * n_arr[mask])
    return vals

x_n = discrete_base(n)

# масштабування амплітуди дискретного сигналу: y[n] = a * x[n]
disc_scale = 1.8
y_disc_scaled = disc_scale * x_n

# дискретний реверс за часом: y[n] = x[-n]
y_disc_rev = discrete_base(-n)

# дискретний зсув у часі: y[n] = x[n - N]
n_shift_delay = 3
n_shift_adv = -3
y_disc_delay = discrete_base(n - n_shift_delay)
y_disc_adv = discrete_base(n - n_shift_adv)

# децимація (downsampling з коефіцієнтом M = 2)
decim_factor = 2
n_decim = np.arange(n[0] // decim_factor, n[-1] // decim_factor + 1)
y_decim = discrete_base(decim_factor * n_decim)

# інтерполяція / upsampling (розширення з коефіцієнтом L = 2 та лінійною інтерполяцією)
upsample_factor = 2
n_up = np.arange(n[0] * upsample_factor, n[-1] * upsample_factor + 1)
# лінійна інтерполяція відліків
y_up_interp = np.interp(n_up / upsample_factor, n, x_n)

# другий дискретний сигнал для операцій накладання
x2_n = 0.5 * np.cos(0.25 * np.pi * n)

# дискретне додавання: y[n] = x1[n] + x2[n]
y_disc_add = x_n + x2_n

# дискретне множення: y[n] = x1[n] * x2[n]
y_disc_mul = x_n * x2_n

print("\n=== ДИСКРЕТНІ СИГНАЛИ ===")
print(f"Діапазон відліків n: [{n[0]}, {n[-1]}], всього {len(n)} відліків")
print(f"Максимальне значення x[n]: {np.max(x_n):.4f}")
print(f"Максимум масштабованого y[n] (a={disc_scale}): {np.max(y_disc_scaled):.4f}")
print(f"Кількість відліків після децимації (M={decim_factor}): {len(y_decim)}")
print(f"Кількість відліків після інтерполяції (L={upsample_factor}): {len(y_up_interp)}")
print(f"Сумарна енергія сигналу Sum(x[n]^2): {np.sum(x_n**2):.4f}")

# графік 4: дискретне масштабування та дискретний реверс
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.stem(n, x_n, linefmt="b-", markerfmt="bo", basefmt="k-", label="x[n]")
plt.stem(n, y_disc_scaled, linefmt="r--", markerfmt="r^", basefmt="k-", label=f"y[n] = {disc_scale}*x[n]")
plt.title("Масштабування амплітуди дискретного сигналу")
plt.xlabel("Індекс відліку n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.stem(n, x_n, linefmt="b-", markerfmt="bo", basefmt="k-", label="x[n]")
plt.stem(n, y_disc_rev, linefmt="m--", markerfmt="ms", basefmt="k-", label="y[n] = x[-n]")
plt.title("Реверс дискретного сигналу за часом")
plt.xlabel("Індекс відліку n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# графік 5: дискретний часовий зсув
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.stem(n, x_n, linefmt="b-", markerfmt="bo", basefmt="k-", label="x[n]")
plt.stem(n, y_disc_delay, linefmt="g--", markerfmt="g^", basefmt="k-", label=f"Затримка: x[n - {n_shift_delay}]")
plt.title(f"Зсув дискретного сигналу (затримка на N={n_shift_delay})")
plt.xlabel("Індекс відліку n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.stem(n, x_n, linefmt="b-", markerfmt="bo", basefmt="k-", label="x[n]")
plt.stem(n, y_disc_adv, linefmt="c--", markerfmt="cv", basefmt="k-", label=f"Випередження: x[n + {abs(n_shift_adv)}]")
plt.title(f"Зсув дискретного сигналу (випередження на N={abs(n_shift_adv)})")
plt.xlabel("Індекс відліку n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# графік 6: децимація та інтерполяція дискретного сигналу
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.stem(n, x_n, linefmt="b-", markerfmt="bo", basefmt="k-", label="x[n] початковий")
plt.stem(n_decim, y_decim, linefmt="r--", markerfmt="rs", basefmt="k-", label=f"Децимація (M={decim_factor})")
plt.title("Децимація: видалення відліків x[M*m]")
plt.xlabel("Індекс m")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.stem(n, x_n, linefmt="b:", markerfmt="bo", basefmt="k-", label="x[n] вихідний")
plt.stem(n_up, y_up_interp, linefmt="g-", markerfmt="g.", basefmt="k-", label=f"Інтерполяція (L={upsample_factor})")
plt.title("Інтерполяція: відновлення проміжних значень")
plt.xlabel("Індекс відліку")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# графік 7: накладання дискретних сигналів (додавання та множення)
plt.figure(figsize=(11, 4.5))
plt.subplot(1, 2, 1)
plt.stem(n, x_n, linefmt="b:", markerfmt="bo", basefmt="k-", label="x1[n]")
plt.stem(n, x2_n, linefmt="gray", markerfmt="none", basefmt="k-", label="x2[n] гармонічний")
plt.stem(n, y_disc_add, linefmt="r--", markerfmt="r^", basefmt="k-", label="x1[n] + x2[n]")
plt.title("Додавання дискретних сигналів")
plt.xlabel("Індекс відліку n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.stem(n, x_n, linefmt="b:", markerfmt="bo", basefmt="k-", label="x1[n]")
plt.stem(n, y_disc_mul, linefmt="g--", markerfmt="gs", basefmt="k-", label="x1[n] * x2[n]")
plt.title("Множення (модуляція) дискретних сигналів")
plt.xlabel("Індекс відліку n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
