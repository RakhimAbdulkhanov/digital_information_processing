import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# параметри дискретизації та часовий вектор
n = 1024
dt = 0.25
t = np.arange(n) * dt
omega0 = 6.0

# 1. модельний сигнал з двома періодичними компонентами
# t1 = 32 с (f1 = 0.03125 гц), t2 = 64 с (f2 = 0.015625 гц)
x = np.sin(2.0 * np.pi * t / 32.0) + np.sin(2.0 * np.pi * t / 64.0)

# графік модельного часового ряду
plt.figure(figsize=(10, 4))
plt.plot(t, x, color="#1f77b4", linewidth=1.5)
plt.title("Модельний часовий ряд x(t) = sin(2*pi*t/32) + sin(2*pi*t/64)")
plt.xlabel("Час t (с)")
plt.ylabel("Амплітуда x(t)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 2. материнський вейвлет морле
eta = np.linspace(-4.0, 4.0, 300)
psi_morlet = np.pi**(-0.25) * np.exp(1j * omega0 * eta) * np.exp(-eta**2 / 2.0)

plt.figure(figsize=(10, 4.5))
plt.plot(eta, psi_morlet.real, color="#1f77b4", linewidth=1.8, label="Дійсна частина Re(psi)")
plt.plot(eta, psi_morlet.imag, color="#d62728", linewidth=1.8, linestyle="--", label="Уявна частина Im(psi)")
plt.plot(eta, np.abs(psi_morlet), color="#2ca02c", linewidth=1.2, linestyle=":", label="Обвідна (гаусіан)")
plt.title("Комплексний материнський вейвлет Морле (omega0 = 6)")
plt.xlabel("Безрозмірний час eta")
plt.ylabel("Амплітуда psi(eta)")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 3. вейвлет мексиканський капелюх та тестова функція
def mexican_hat(val):
    return (2.0 / (np.sqrt(3.0) * np.pi**0.25)) * (1.0 - val**2) * np.exp(-val**2 / 2.0)

t_mh = np.linspace(-5.0, 5.0, 400)
psi_mh = mexican_hat(t_mh)

x_test = np.linspace(0.0, 30.0, 600)
dx_test = x_test[1] - x_test[0]
f_test = mexican_hat(x_test - 10.0) + mexican_hat((x_test - 20.0) / 2.0)

# фур'є-спектр тестової функції
f_test_fft = np.fft.rfft(f_test) * dx_test
freqs_test = np.fft.rfftfreq(len(f_test), d=dx_test)

plt.figure(figsize=(12, 4.2))
plt.subplot(1, 3, 1)
plt.plot(t_mh, psi_mh, color="#9467bd", linewidth=1.8)
plt.title("Вейвлет 'Мексиканський капелюх'")
plt.xlabel("Час t")
plt.ylabel("psi(t)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 3, 2)
plt.plot(x_test, f_test, color="#e377c2", linewidth=1.8)
plt.title("Тестова функція f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 3, 3)
plt.plot(freqs_test, np.abs(f_test_fft), color="#ff7f0e", linewidth=1.8)
plt.title("Фур'є-спектр |F(f)|")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Амплітуда")
plt.xlim(0, 1.0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# 4. неперервне вейвлет-перетворення (cwt) для ряду x(t)
# набір масштабів за двійковою шкалою
s0 = 2.0 * dt
dj = 0.25
j_max = int(np.log2(n * dt / s0) / dj)
scales = s0 * 2.0**(np.arange(j_max + 1) * dj)

# еквівалентний фур'є-період для вейвлету морле
fourier_factor = (4.0 * np.pi) / (omega0 + np.sqrt(2.0 + omega0**2))
fourier_period = scales * fourier_factor

# обчислення коефіцієнтів вейвлет-перетворення як інтегральної згортки
w = np.zeros((len(scales), n), dtype=complex)
for i, s in enumerate(scales):
    eta_scaled = (t[:, None] - t) / s
    psi_scaled = np.pi**(-0.25) * np.exp(1j * omega0 * eta_scaled) * np.exp(-eta_scaled**2 / 2.0)
    psi_scaled = psi_scaled / np.sqrt(s)
    w[i, :] = np.sum(x * psi_scaled, axis=1) * dt

# спектральна густина потужності (скалограма)
power = np.abs(w)**2

# 2d скалограма
plt.figure(figsize=(11, 5.5))
levels = np.linspace(0, power.max(), 50)
cf = plt.contourf(t, fourier_period, power, levels=levels, cmap="jet", extend="both")
cbar = plt.colorbar(cf, pad=0.02)
cbar.set_label("Спектральна потужність |W(s, t)|^2", fontsize=11)
plt.axhline(32.0, color="white", linestyle="--", linewidth=1.5, alpha=0.8, label="Період T1 = 32 с")
plt.axhline(64.0, color="yellow", linestyle="--", linewidth=1.5, alpha=0.8, label="Період T2 = 64 с")
plt.yscale("log")
plt.ylim(fourier_period.min(), fourier_period.max())
plt.title("2D Скалограма неперервного вейвлет-перетворення (вейвлет Морле)")
plt.xlabel("Час t (с)", fontsize=11)
plt.ylabel("Період коливань T (с)", fontsize=11)
plt.legend(loc="upper right", framealpha=0.8)
plt.tight_layout()
plt.show()

# 3d скалограма
fig = plt.figure(figsize=(12, 6.5))
ax = fig.add_subplot(111, projection="3d")
t_mesh, period_mesh = np.meshgrid(t, fourier_period)

surf = ax.plot_surface(t_mesh, np.log2(period_mesh), power, cmap="jet", edgecolor="none", alpha=0.92)
ax.set_xlabel("Час t (с)", fontsize=10, labelpad=8)
ax.set_ylabel("log2(Період T)", fontsize=10, labelpad=8)
ax.set_zlabel("Потужність |W|^2", fontsize=10, labelpad=8)
ax.set_title("3D Скалограма модельного сигналу", fontsize=12)
ax.view_init(elev=32, azim=48)
cbar_3d = fig.colorbar(surf, shrink=0.55, aspect=8, pad=0.08)
cbar_3d.set_label("Потужність |W|^2", fontsize=10)
plt.tight_layout()
plt.show()

# пошук пікових періодів на скалограмі
mean_power = np.mean(power[:, 200:800], axis=1)
peak_indices = np.argsort(mean_power)[-2:]
detected_periods = np.sort(fourier_period[peak_indices])

# консольний вивід розрахованих параметрів
print(f"Параметри сигналу: N = {n}, dt = {dt} c, T_total = {n * dt:.1f} c, Fs = {1.0 / dt:.1f} Гц")
print(f"Теоретичні гармоніки: T1 = 32.00 с (f1 = 0.03125 Гц), T2 = 64.00 с (f2 = 0.015625 Гц)")
print(f"Параметри CWT: s0 = {s0:.2f}, dj = {dj:.2f}, кількість масштабів J+1 = {len(scales)}")
print(f"Діапазон масштабів s: від {scales.min():.2f} до {scales.max():.2f}")
print(f"Діапазон періодів Фур'є: від {fourier_period.min():.2f} с до {fourier_period.max():.2f} с")
print(f"Виявлені пікові періоди за скалограмою: {detected_periods[0]:.2f} с та {detected_periods[1]:.2f} с")
