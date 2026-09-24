import math
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. залежність власної інформації I(E) від ймовірності P(E)
# -------------------------------------------------------------
p_vals = np.linspace(0.001, 1.0, 1000)
info_vals = -np.log2(p_vals)

plt.figure(figsize=(9, 4.8))
plt.plot(p_vals, info_vals, color="#1f77b4", linewidth=2.2, label=r"$I(E) = -\log_2 P(E)$")
plt.axvline(0.5, color="#d62728", linestyle="--", alpha=0.7, label="P(E) = 0.5 (I = 1 біт)")
plt.axhline(1.0, color="#d62728", linestyle="--", alpha=0.7)
plt.scatter([0.5, 1.0], [1.0, 0.0], color="#d62728", s=40, zorder=5)
plt.annotate("1 біт (підкидання монети)", xy=(0.5, 1.0), xytext=(0.55, 2.0),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))
plt.title("Кількість інформації I(E) залежно від ймовірності появи події P(E)")
plt.xlabel("Ймовірність появи події P(E)")
plt.ylabel("Кількість інформації I(E) (біт)")
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 10.0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 2. вибірка випадкових чисел 0-9 для N=100
# -------------------------------------------------------------
np.random.seed(42)
n100 = 100
a100 = np.random.randint(0, 10, size=n100)

counts100 = np.bincount(a100, minlength=10)
probs100 = counts100 / n100
infos100 = np.zeros(10)
for i, p in enumerate(probs100):
    if p > 0:
        infos100[i] = -np.log2(p)

print("--- Вибірка N = 100 елементів (числа 0-9) ---")
print("Згенерований масив:", a100.tolist())
print("Кількість появ кожного числа:", counts100.tolist())
print("Ймовірності P(E) та інформація I(E):")
for i in range(10):
    print(f"Число {i}: кількість = {counts100[i]:2d}, P = {probs100[i]:.3f}, I = {infos100[i]:.3f} біт")

# графік ймовірностей
plt.figure(figsize=(9, 4.2))
plt.bar(range(10), probs100, color="#2ca02c", edgecolor="black", alpha=0.85)
plt.axhline(0.1, color="red", linestyle="--", label="Теоретична ймовірність P = 0.1")
plt.title("Оцінка ймовірності випадання чисел 0-9 (N = 100)")
plt.xlabel("Число (символ алфавіту джерела)")
plt.ylabel("Ймовірність P(E)")
plt.xticks(range(10))
plt.ylim(0, max(probs100) + 0.05)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# графік інформаційної міри кожного числа
plt.figure(figsize=(9, 4.2))
plt.bar(range(10), infos100, color="#1f77b4", edgecolor="black", alpha=0.85)
plt.axhline(-np.log2(0.1), color="red", linestyle="--", label=r"Теоретична інформація $-\log_2(0.1) \approx 3.322$ біт")
plt.title("Інформаційна міра подій появи чисел (N = 100)")
plt.xlabel("Число")
plt.ylabel("I(E) = -log2(P(E)) (біт)")
plt.xticks(range(10))
plt.ylim(0, max(infos100) + 0.8)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 3. залежність ентропії шеннона від розміру вибірки N
# -------------------------------------------------------------
sample_sizes = [50, 100, 200, 500, 1000, 2000, 3000, 5000]
entropies = []
redundancies = []
h_max_10 = np.log2(10)

print("\n--- Дослідження ентропії Шеннона для різних розмірів вибірки ---")
print(f"Теоретичний максимум H_max = log2(10) = {h_max_10:.4f} біт")

for n in sample_sizes:
    arr = np.random.randint(0, 10, size=n)
    counts = np.bincount(arr, minlength=10)
    p = counts / n
    # шеннонівська ентропія
    h = -np.sum([pi * np.log2(pi) for pi in p if pi > 0])
    # надлишковість джерела
    r = 1.0 - (h / h_max_10)
    entropies.append(h)
    redundancies.append(r)
    print(f"N = {n:4d} | Ентропія H = {h:.4f} біт | Надлишковість R = {r * 100:.2f}%")

plt.figure(figsize=(9, 4.8))
plt.plot(sample_sizes, entropies, marker="o", color="#d62728", linewidth=2, label="Емпірична ентропія H(N)")
plt.axhline(h_max_10, color="black", linestyle="--", linewidth=1.5, label=f"H_max = log2(10) = {h_max_10:.4f} біт")
plt.title("Залежність ентропії джерела від розміру вибірки N")
plt.xlabel("Розмір вибірки N")
plt.ylabel("Ентропія Шеннона H (біт/символ)")
plt.ylim(3.1, 3.35)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 4. ентропія сигналів із різними законами розподілу
# -------------------------------------------------------------
# бінарне джерело та графік H(p)
p_bin = np.linspace(0.0001, 0.9999, 500)
h_bin = -p_bin * np.log2(p_bin) - (1.0 - p_bin) * np.log2(1.0 - p_bin)
# функція невизначеності одного символу H1(p) = -p*log2(p)
h1_single = -p_bin * np.log2(p_bin)

plt.figure(figsize=(10, 4.5))
plt.subplot(1, 2, 1)
plt.plot(p_bin, h_bin, color="#9467bd", linewidth=2.2)
plt.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
plt.scatter([0.5], [1.0], color="#d62728", s=40, zorder=5)
plt.title("Ентропія бінарного джерела H(p)")
plt.xlabel("Ймовірність p")
plt.ylabel("Ентропія (біт)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(p_bin, h1_single, color="#e377c2", linewidth=2.2)
p_max_h1 = 1.0 / np.e
h1_max = -p_max_h1 * np.log2(p_max_h1)
plt.axvline(p_max_h1, color="gray", linestyle="--", alpha=0.7)
plt.scatter([p_max_h1], [h1_max], color="#d62728", s=40, zorder=5)
plt.annotate(f"Максимум (p = 1/e ≈ {p_max_h1:.3f})", xy=(p_max_h1, h1_max),
             xytext=(p_max_h1 + 0.08, h1_max - 0.08),
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))
plt.title("Часткова невизначеність H1(p) = -p*log2(p)")
plt.xlabel("Ймовірність p")
plt.ylabel("H1(p) (біт)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# порівняння розподілів (рівномірний, нормальний, скошений/експоненційний)
n_samples = 10000
num_bins = 16
h_max_16 = np.log2(num_bins)

# 1) рівномірний
s_uniform = np.random.randint(0, num_bins, size=n_samples)
p_unif = np.bincount(s_uniform, minlength=num_bins) / n_samples
h_unif = -np.sum([p * np.log2(p) for p in p_unif if p > 0])
r_unif = 1.0 - h_unif / h_max_16

# 2) нормальний (дискретизований)
s_norm_raw = np.random.normal(loc=7.5, scale=2.0, size=n_samples)
s_norm = np.clip(np.round(s_norm_raw).astype(int), 0, num_bins - 1)
p_norm = np.bincount(s_norm, minlength=num_bins) / n_samples
h_norm = -np.sum([p * np.log2(p) for p in p_norm if p > 0])
r_norm = 1.0 - h_norm / h_max_16

# 3) скошений (експоненційний)
s_skew_raw = np.random.exponential(scale=2.5, size=n_samples)
s_skew = np.clip(np.floor(s_skew_raw).astype(int), 0, num_bins - 1)
p_skew = np.bincount(s_skew, minlength=num_bins) / n_samples
h_skew = -np.sum([p * np.log2(p) for p in p_skew if p > 0])
r_skew = 1.0 - h_skew / h_max_16

print("\n--- Порівняння ентропії для різних розподілів (M = 16 рівнів) ---")
print(f"Максимальна ентропія H_max = log2(16) = {h_max_16:.4f} біт")
print(f"1. Рівномірний розподіл:   H = {h_unif:.4f} біт | Надлишковість R = {r_unif * 100:.2f}%")
print(f"2. Нормальний розподіл:    H = {h_norm:.4f} біт | Надлишковість R = {r_norm * 100:.2f}%")
print(f"3. Скошений розподіл:      H = {h_skew:.4f} біт | Надлишковість R = {r_skew * 100:.2f}%")

plt.figure(figsize=(12, 4.2))
plt.subplot(1, 3, 1)
plt.bar(range(num_bins), p_unif, color="#1f77b4", edgecolor="black", alpha=0.8)
plt.title(f"Рівномірний\nH = {h_unif:.3f} біт, R = {r_unif*100:.1f}%")
plt.xlabel("Рівень квантування")
plt.ylabel("Ймовірність")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.subplot(1, 3, 2)
plt.bar(range(num_bins), p_norm, color="#2ca02c", edgecolor="black", alpha=0.8)
plt.title(f"Нормальний\nH = {h_norm:.3f} біт, R = {r_norm*100:.1f}%")
plt.xlabel("Рівень квантування")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.subplot(1, 3, 3)
plt.bar(range(num_bins), p_skew, color="#ff7f0e", edgecolor="black", alpha=0.8)
plt.title(f"Скошений\nH = {h_skew:.3f} біт, R = {r_skew*100:.1f}%")
plt.xlabel("Рівень квантування")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 5. оцінка ентропії цифрових зображень на основі гістограм
# -------------------------------------------------------------
# створення трьох тестових 8-бітних зображень 128x128 (256 рівнів)
h_img, w_img = 128, 128
h_max_img = 8.0 # log2(256) біт на піксель

# а) випадковий шум (максимальна невизначеність)
img_noise = np.random.randint(0, 256, size=(h_img, w_img), dtype=np.uint8)

# б) плавний градієнт зі структурою
x_coords = np.linspace(0, 255, w_img)
y_coords = np.linspace(0, 255, h_img)
xx, yy = np.meshgrid(x_coords, y_coords)
img_gradient = np.clip((xx + yy) / 2.0, 0, 255).astype(np.uint8)

# в) низькоентропійне зображення (контрастні смуги / фігури)
img_pattern = np.zeros((h_img, w_img), dtype=np.uint8)
img_pattern[:64, :] = 50
img_pattern[64:, :] = 200

def calc_image_entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    p = hist / img.size
    h = -np.sum([pk * np.log2(pk) for pk in p if pk > 0])
    r = 1.0 - h / h_max_img
    return h, r, p

h_img1, r_img1, p_img1 = calc_image_entropy(img_noise)
h_img2, r_img2, p_img2 = calc_image_entropy(img_gradient)
h_img3, r_img3, p_img3 = calc_image_entropy(img_pattern)

print("\n--- Ентропійна оцінка 8-бітових зображень (M = 256 рівнів) ---")
print(f"1. Рівномірний шум:   H = {h_img1:.4f} біт/піксель | Надлишковість R = {r_img1 * 100:.2f}%")
print(f"2. Плавний градієнт:  H = {h_img2:.4f} біт/піксель | Надлишковість R = {r_img2 * 100:.2f}%")
print(f"3. Двохрівневий узор: H = {h_img3:.4f} біт/піксель | Надлишковість R = {r_img3 * 100:.2f}%")

plt.figure(figsize=(12, 6.5))
# зображення
plt.subplot(2, 3, 1)
plt.imshow(img_noise, cmap="gray", vmin=0, vmax=255)
plt.title(f"Шумове зображення\nH = {h_img1:.2f} біт/пікс")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(img_gradient, cmap="gray", vmin=0, vmax=255)
plt.title(f"Градієнтне зображення\nH = {h_img2:.2f} біт/пікс")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(img_pattern, cmap="gray", vmin=0, vmax=255)
plt.title(f"Двохрівневий узор\nH = {h_img3:.2f} біт/пікс")
plt.axis("off")

# гістограми яскравості
plt.subplot(2, 3, 4)
plt.plot(p_img1, color="#1f77b4")
plt.fill_between(range(256), p_img1, color="#1f77b4", alpha=0.3)
plt.title(f"Гістограма (R = {r_img1*100:.1f}%)")
plt.xlabel("Яскравість (0..255)")
plt.ylabel("Ймовірність P(k)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(2, 3, 5)
plt.plot(p_img2, color="#2ca02c")
plt.fill_between(range(256), p_img2, color="#2ca02c", alpha=0.3)
plt.title(f"Гістограма (R = {r_img2*100:.1f}%)")
plt.xlabel("Яскравість (0..255)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(2, 3, 6)
plt.plot(p_img3, color="#ff7f0e")
plt.fill_between(range(256), p_img3, color="#ff7f0e", alpha=0.3)
plt.title(f"Гістограма (R = {r_img3*100:.1f}%)")
plt.xlabel("Яскравість (0..255)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
