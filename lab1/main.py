import numpy as np
import matplotlib.pyplot as plt
import math
import os

# створюєм папку для графіків
plots_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plots_dir, exist_ok=True)

# налаштування шрифтів шоб графіки виглядали чітко
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

print("=" * 60)
print("Лабораторна робота №1: Детерміновані сигнали для ЦОІ")
print("Виконав: Абдулханов Абдул-Рахім (ФеП-33)")
print("Викладач: проф. Половинко І. І.")
print("=" * 60)

# -------------------------------------------------------------
# 1. Прямокутний імпульс (rect) та sinc
# -------------------------------------------------------------
print("\n[1] Прямокутний імпульс Rect(t) та Фур'є-образ sinc(f)")
print("- Формула: rect(t) = 1 при |t| <= tau/2, інакше 0")
print("- Фур'є-образ: F{rect(t)} = tau * sinc(tau * f) = sin(pi * tau * f) / (pi * f)")
print("- Практичне значення: тактові імпульси, двійкові біти в шинах даних, стробування")

def rect_func(t, width=1.0):
    return np.where(np.abs(t) <= width / 2.0, 1.0, 0.0)

def sinc_func(f, width=1.0):
    return np.sinc(width * f)

t_rect = np.linspace(-3.0, 3.0, 1000)
f_rect = np.linspace(-4.0, 4.0, 1000)
tau = 1.0

y_rect = rect_func(t_rect, tau)
y_sinc = tau * sinc_func(f_rect, tau)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(t_rect, y_rect, color="#1f77b4", linewidth=2)
ax1.set_title("Прямокутний сигнал rect(t) (tau = 1.0)")
ax1.set_xlabel("Час t (с)")
ax1.set_ylabel("Амплітуда")
ax1.set_ylim(-0.2, 1.3)
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(f_rect, y_sinc, color="#d62728", linewidth=2)
ax2.set_title("Фур'є-образ sinc(f) = sin(pi*f)/(pi*f)")
ax2.set_xlabel("Частота f (Гц)")
ax2.set_ylabel("Спектральна щільність")
ax2.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "1_rect_and_sinc.png"), dpi=200)
plt.close()

# окремий детальний графік sinc функції з додатка методички
plt.figure(figsize=(8.5, 4.8))
plt.plot(f_rect, y_sinc, color="#d62728", linewidth=2)
plt.title("Sinc функція - перетворення Фур'є прямокутного сигналу")
plt.xlabel("Частота f (Гц)")
plt.ylabel("Амплітуда")
plt.text(0, 0.8, f"sinc(f) = sin(pi*{tau}*f) / (pi*{tau}*f)",
         fontsize=11, ha="center", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc"))
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim(-4, 4)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "1b_sinc_detailed.png"), dpi=200)
plt.close()

print("- Перші нулі спектра: f = +-1.0 Гц (обернено пропорційно тривалості tau)")

# -------------------------------------------------------------
# 2. Гармонічний сигнал (косинус) та дельта-піки
# -------------------------------------------------------------
print("\n[2] Гармонічний косинусоїдальний сигнал cos(2*pi*f0*t)")
print("- Формула: s(t) = cos(2*pi*f0*t)")
print("- Фур'є-образ: F{cos} = 0.5 * delta(f + f0) + 0.5 * delta(f - f0)")
print("- Практичне значення: опорні коливання гетеродина, несучі частоти в радіозв'язку")

f0 = 2.0
t_cos = np.linspace(-2.0, 2.0, 1000)
y_cos = np.cos(2 * np.pi * f0 * t_cos)

# розрахунок амплітудного спектра через швидке перетворення фур'є
N_fft = 2048
dt = t_cos[1] - t_cos[0]
y_cos_long = np.cos(2 * np.pi * f0 * np.linspace(0, 10, N_fft))
spectrum_cos = np.abs(np.fft.fftshift(np.fft.fft(y_cos_long))) / N_fft
freq_cos = np.fft.fftshift(np.fft.fftfreq(N_fft, d=10/N_fft))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(t_cos, y_cos, color="#1f77b4", linewidth=2)
ax1.set_title(f"Косинусоїдальний сигнал (f0 = {f0} Гц)")
ax1.set_xlabel("Час t (с)")
ax1.set_ylabel("Амплітуда")
ax1.grid(True, linestyle="--", alpha=0.6)

# обмежуєм частоту шоб піки було видно крупно
mask = np.abs(freq_cos) <= 6.0
ax2.plot(freq_cos[mask], spectrum_cos[mask], color="#d62728", linewidth=2)
ax2.set_title("Спектр: пара симетричних дельта-піків (+-f0)")
ax2.set_xlabel("Частота f (Гц)")
ax2.set_ylabel("Амплітуда спектра")
ax2.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "2_cosine_spectrum.png"), dpi=200)
plt.close()
print(f"- Спектр локалізований: енергія сконцентрована точно на +- {f0} Гц")

# -------------------------------------------------------------
# 3. Дельта-функція Дірака та рівномірний спектр
# -------------------------------------------------------------
print("\n[3] Дельта-функція Дірака delta(t) та нескінченний спектр")
print("- Формула: delta(t) = нескінченність при t=0, інтеграл = 1")
print("- Фур'є-образ: F{delta(t)} = 1 (рівномірний неперервний спектр на всіх частотах)")
print("- Практичне значення: вимірювання імпульсної характеристики h(t) лінійних фільтрів")

t_delta = np.linspace(-1.0, 1.0, 1000)
# числова апроксимація вузьким гаусовим імпульсом
sigma = 0.015
y_delta = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (t_delta / sigma)**2)

f_delta = np.linspace(-20.0, 20.0, 1000)
# аналітичний спектр дельта-функції дорівнює одиниці
spectrum_delta = np.ones_like(f_delta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(t_delta, y_delta, color="#1f77b4", linewidth=2)
ax1.set_title("Дельта-імпульс delta(t) (апроксимація)")
ax1.set_xlabel("Час t (с)")
ax1.set_ylabel("Амплітуда")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(f_delta, spectrum_delta, color="#d62728", linewidth=2)
ax2.set_title("Фур'є-образ F{delta(t)} = 1 (білий спектр)")
ax2.set_xlabel("Частота f (Гц)")
ax2.set_ylabel("Спектральна щільність")
ax2.set_ylim(0.0, 1.5)
ax2.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "3_dirac_delta.png"), dpi=200)
plt.close()
print("- Властивість: нульова тривалість у часі дає нескінченно широку смугу частот")

# -------------------------------------------------------------
# 4. Функція знаку signum (sgn)
# -------------------------------------------------------------
print("\n[4] Функція знаку sgn(t)")
print("- Формула: sgn(t) = +1 (t>0), 0 (t=0), -1 (t<0)")
print("- Фур'є-образ: F{sgn(t)} = 1 / (j * pi * f), модуль: |F| = 1 / (pi * |f|)")
print("- Практичне значення: перетворення Гільберта, фазові детектори, формування SSB сигналів")

t_sgn = np.linspace(-3.0, 3.0, 1000)
y_sgn = np.sign(t_sgn)

f_sgn = np.linspace(0.1, 5.0, 500)
mag_sgn = 1.0 / (np.pi * f_sgn)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.step(t_sgn, y_sgn, where="mid", color="#1f77b4", linewidth=2)
ax1.set_title("Функція signum: sgn(t)")
ax1.set_xlabel("Час t (с)")
ax1.set_ylabel("Амплітуда")
ax1.set_ylim(-1.4, 1.4)
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(f_sgn, mag_sgn, color="#d62728", linewidth=2)
ax2.set_title("Модуль спектра: |F{sgn(t)}| = 1 / (pi * f)")
ax2.set_xlabel("Частота f (Гц)")
ax2.set_ylabel("Амплітуда спектра")
ax2.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "4_signum_function.png"), dpi=200)
plt.close()
print("- Особливість: неінтегровна функція, енергія спадає гіперболічно ~ 1/f")

# -------------------------------------------------------------
# 5. Трикутний імпульс (tri) та sinc^2
# -------------------------------------------------------------
print("\n[5] Трикутний імпульс tri(t) та Фур'є-образ sinc^2(f)")
print("- Формула: tri(t) = 1 - |t|/tau при |t| <= tau, інакше 0")
print("- Фур'є-образ: F{tri(t)} = tau * sinc^2(tau * f)")
print("- Практичне значення: віконна фільтрація (вікно Бартлетта), інтерполяція, згладжування")

tau_tri = 1.0
t_tri = np.linspace(-2.5, 2.5, 1000)
y_tri = np.maximum(0.0, 1.0 - np.abs(t_tri) / tau_tri)

f_tri = np.linspace(-4.0, 4.0, 1000)
y_tri_fourier = tau_tri * (np.sinc(tau_tri * f_tri))**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(t_tri, y_tri, color="#1f77b4", linewidth=2)
ax1.set_title("Трикутний імпульс tri(t)")
ax1.set_xlabel("Час t (с)")
ax1.set_ylabel("Амплітуда")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(f_tri, y_tri_fourier, color="#d62728", linewidth=2)
ax2.set_title("Фур'є-образ: sinc^2(f)")
ax2.set_xlabel("Частота f (Гц)")
ax2.set_ylabel("Спектральна щільність")
ax2.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "5_triangular_sinc2.png"), dpi=200)
plt.close()
print("- Порівняння з rect: бічні пелюстки sinc^2 спадають швидше (~1/f^2 проти 1/f у rect)")

# -------------------------------------------------------------
# 6. Функція відліків sinc(t) та прямокутний спектр
# -------------------------------------------------------------
print("\n[6] Функція відліків sinc(t) та прямокутний фільтр")
print("- Формула: sinc(t) = sin(pi * t) / (pi * t)")
print("- Фур'є-образ: F{sinc(2*B*t)} = (1 / 2B) * rect(f / 2B)")
print("- Практичне значення: теорема Найквіста-Котельникова (ідеальна дискретизація), brick-wall ФНЧ")

t_sinc_time = np.linspace(-5.0, 5.0, 1000)
y_sinc_time = np.sinc(t_sinc_time)

f_sinc_spec = np.linspace(-2.0, 2.0, 1000)
y_rect_spec = np.where(np.abs(f_sinc_spec) <= 0.5, 1.0, 0.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(t_sinc_time, y_sinc_time, color="#1f77b4", linewidth=2)
ax1.set_title("Сигнал sinc(t) у часовій області")
ax1.set_xlabel("Час t (с)")
ax1.set_ylabel("Амплітуда")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(f_sinc_spec, y_rect_spec, color="#d62728", linewidth=2)
ax2.set_title("Спектр: ідеальний прямокутний діапазон |f| <= 0.5")
ax2.set_xlabel("Частота f (Гц)")
ax2.set_ylabel("Амплітуда спектра")
ax2.set_ylim(-0.2, 1.3)
ax2.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "6_sinc_time_rect_spec.png"), dpi=200)
plt.close()
print("- Властивість дуальності: прямокутник у часі дає sinc у частоті, а sinc у часі дає прямокутник у частоті")

print("\n" + "=" * 60)
print("Всі 6 сигналів успішно змодельовано та збережено у папку plots/")
print("=" * 60)
