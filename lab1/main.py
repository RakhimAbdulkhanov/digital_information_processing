import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. прямокутний імпульс та sinc функція
# -------------------------------------------------------------
def rect_func(t, width=1.0):
    return np.where(np.abs(t) <= width / 2.0, 1.0, 0.0)

def sinc_func(f, width=1.0):
    return np.sinc(width * f)

t_rect = np.linspace(-3.0, 3.0, 1000)
f_rect = np.linspace(-4.0, 4.0, 1000)
tau = 1.0

y_rect = rect_func(t_rect, tau)
y_sinc = tau * sinc_func(f_rect, tau)

plt.figure(figsize=(11, 4.2))
plt.subplot(1, 2, 1)
plt.plot(t_rect, y_rect, color="#1f77b4", linewidth=2)
plt.title("Прямокутний сигнал rect(t)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.ylim(-0.2, 1.3)
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(f_rect, y_sinc, color="#d62728", linewidth=2)
plt.title("Фур'є-образ: sinc(f)")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Спектральна щільність")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# окремий графік sinc з анотацією
plt.figure(figsize=(8.5, 4.8))
plt.plot(f_rect, y_sinc, color="#d62728", linewidth=2)
plt.title("Sinc функція - спектр прямокутного імпульсу")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Амплітуда")
plt.text(0, 0.8, f"sinc(f) = sin(pi*{tau}*f) / (pi*{tau}*f)",
         fontsize=11, ha="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc"))
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(-4, 4)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 2. гармонічний сигнал (косинус)
# -------------------------------------------------------------
f0 = 2.0
t_cos = np.linspace(-2.0, 2.0, 1000)
y_cos = np.cos(2 * np.pi * f0 * t_cos)

# спектр через швидке перетворення фур'є
n_fft = 2048
y_cos_long = np.cos(2 * np.pi * f0 * np.linspace(0, 10, n_fft))
spectrum_cos = np.abs(np.fft.fftshift(np.fft.fft(y_cos_long))) / n_fft
freq_cos = np.fft.fftshift(np.fft.fftfreq(n_fft, d=10 / n_fft))

plt.figure(figsize=(11, 4.2))
plt.subplot(1, 2, 1)
plt.plot(t_cos, y_cos, color="#1f77b4", linewidth=2)
plt.title(f"Косинусоїдальний сигнал (f0 = {f0} Гц)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
mask = np.abs(freq_cos) <= 6.0
plt.plot(freq_cos[mask], spectrum_cos[mask], color="#d62728", linewidth=2)
plt.title("Спектр: піки на частотах +- f0")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 3. дельта-функція дірака
# -------------------------------------------------------------
t_delta = np.linspace(-1.0, 1.0, 1000)
# апроксимація гаусовим куполом
sigma = 0.015
y_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (t_delta / sigma)**2)

f_delta = np.linspace(-20.0, 20.0, 1000)
spectrum_delta = np.ones_like(f_delta)

plt.figure(figsize=(11, 4.2))
plt.subplot(1, 2, 1)
plt.plot(t_delta, y_delta, color="#1f77b4", linewidth=2)
plt.title("Дельта-імпульс delta(t)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(f_delta, spectrum_delta, color="#d62728", linewidth=2)
plt.title("Фур'є-образ: F = 1 (рівномірний спектр)")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Спектральна щільність")
plt.ylim(0.0, 1.5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 4. функція знаку signum
# -------------------------------------------------------------
t_sgn = np.linspace(-3.0, 3.0, 1000)
y_sgn = np.sign(t_sgn)

f_sgn = np.linspace(0.1, 5.0, 500)
mag_sgn = 1.0 / (np.pi * f_sgn)

plt.figure(figsize=(11, 4.2))
plt.subplot(1, 2, 1)
plt.step(t_sgn, y_sgn, where="mid", color="#1f77b4", linewidth=2)
plt.title("Функція sgn(t)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.ylim(-1.4, 1.4)
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(f_sgn, mag_sgn, color="#d62728", linewidth=2)
plt.title("Модуль спектра: |F| = 1 / (pi * f)")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 5. трикутний імпульс
# -------------------------------------------------------------
tau_tri = 1.0
t_tri = np.linspace(-2.5, 2.5, 1000)
y_tri = np.maximum(0.0, 1.0 - np.abs(t_tri) / tau_tri)

f_tri = np.linspace(-4.0, 4.0, 1000)
y_tri_fourier = tau_tri * (np.sinc(tau_tri * f_tri))**2

plt.figure(figsize=(11, 4.2))
plt.subplot(1, 2, 1)
plt.plot(t_tri, y_tri, color="#1f77b4", linewidth=2)
plt.title("Трикутний імпульс tri(t)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(f_tri, y_tri_fourier, color="#d62728", linewidth=2)
plt.title("Фур'є-образ: sinc^2(f)")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Спектральна щільність")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 6. функція відліків sinc(t)
# -------------------------------------------------------------
t_sinc_time = np.linspace(-5.0, 5.0, 1000)
y_sinc_time = np.sinc(t_sinc_time)

f_sinc_spec = np.linspace(-2.0, 2.0, 1000)
y_rect_spec = np.where(np.abs(f_sinc_spec) <= 0.5, 1.0, 0.0)

plt.figure(figsize=(11, 4.2))
plt.subplot(1, 2, 1)
plt.plot(t_sinc_time, y_sinc_time, color="#1f77b4", linewidth=2)
plt.title("Сигнал sinc(t)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(f_sinc_spec, y_rect_spec, color="#d62728", linewidth=2)
plt.title("Спектр: прямокутний фільтр |f| <= 0.5")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Амплітуда")
plt.ylim(-0.2, 1.3)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
