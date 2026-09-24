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

# окремий детальний графік sinc
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
# 2. гармонічний сигнал (косинус) та дельта-піки спектра
# -------------------------------------------------------------
f0 = 2.0
t_cos = np.linspace(-2.0, 2.0, 1000)
y_cos = np.cos(2 * np.pi * f0 * t_cos)

f_cos = np.linspace(-6.0, 6.0, 1000)
spec_cos_base = np.zeros_like(f_cos)

plt.figure(figsize=(11, 4.2))
plt.subplot(1, 2, 1)
plt.plot(t_cos, y_cos, color="#1f77b4", linewidth=2)
plt.title(f"Косинусоїдальний сигнал (f0 = {f0} Гц)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(f_cos, spec_cos_base, color="#d62728", linewidth=1.5)
# чисті дельта-піки без числового розмиття
plt.vlines([-f0, f0], ymin=0, ymax=0.5, color="#d62728", linewidth=2.5)
plt.annotate("", xy=(-f0, 0.5), xytext=(-f0, 0),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.5, mutation_scale=15))
plt.annotate("", xy=(f0, 0.5), xytext=(f0, 0),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.5, mutation_scale=15))
plt.scatter([-f0, f0], [0.5, 0.5], color="#d62728", s=30, zorder=5)
plt.title("Спектр: дельта-піки на частотах +- f0")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Амплітуда")
plt.ylim(-0.05, 0.65)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 3. дельта-функція дірака та рівномірний спектр
# -------------------------------------------------------------
t_delta = np.linspace(-1.0, 1.0, 1000)
y_delta_base = np.zeros_like(t_delta)

f_delta = np.linspace(-20.0, 20.0, 1000)
spectrum_delta = np.ones_like(f_delta)

plt.figure(figsize=(11, 4.2))
plt.subplot(1, 2, 1)
plt.plot(t_delta, y_delta_base, color="#1f77b4", linewidth=1.5)
# ідеальний дельта-імпульс у точці 0
plt.vlines(0, ymin=0, ymax=1.0, color="#1f77b4", linewidth=2.5)
plt.annotate("", xy=(0, 1.0), xytext=(0, 0),
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2.5, mutation_scale=15))
plt.scatter([0], [1.0], color="#1f77b4", s=30, zorder=5)
plt.title("Дельта-імпульс delta(t)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.ylim(-0.1, 1.25)
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

f_sgn_neg = np.linspace(-4.0, -0.06, 500)
f_sgn_pos = np.linspace(0.06, 4.0, 500)

im_sgn_neg = -1.0 / (np.pi * f_sgn_neg)
im_sgn_pos = -1.0 / (np.pi * f_sgn_pos)

mag_sgn_neg = 1.0 / (np.pi * np.abs(f_sgn_neg))
mag_sgn_pos = 1.0 / (np.pi * np.abs(f_sgn_pos))

plt.figure(figsize=(11, 4.2))
plt.subplot(1, 2, 1)
plt.step(t_sgn, y_sgn, where="mid", color="#1f77b4", linewidth=2)
plt.title("Функція sgn(t)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда")
plt.ylim(-1.4, 1.4)
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
# уявна частина згідно з рис. 14 завдання та модуль спектра
plt.plot(f_sgn_neg, im_sgn_neg, color="#d62728", linestyle="--", linewidth=2, label="Im(F) = -1/(pi*f)")
plt.plot(f_sgn_pos, im_sgn_pos, color="#d62728", linestyle="--", linewidth=2)
plt.plot(f_sgn_neg, mag_sgn_neg, color="#ff7f0e", linewidth=1.8, label="|F| = 1/(pi*|f|)")
plt.plot(f_sgn_pos, mag_sgn_pos, color="#ff7f0e", linewidth=1.8)
plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
plt.axvline(0, color="gray", linestyle=":", alpha=0.5)
plt.title("Спектр: F{sgn(t)} = 1 / (j*pi*f)")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Амплітуда")
plt.ylim(-3.5, 3.5)
plt.legend(loc="upper right", framealpha=0.9)
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
