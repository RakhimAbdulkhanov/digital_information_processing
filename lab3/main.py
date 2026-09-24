import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. взаємна кореляція та енергія сигналів з таблиці 1
# -------------------------------------------------------------
# масиви значень дискретних сигналів
x1_t1 = np.array([4, 2, -1, 3, -2, -6, -5, 4, 5], dtype=float)
x2_t1 = np.array([7, 4, -2, -8, -2, -1, 0, 0, 0], dtype=float)
n_t1 = np.arange(1, len(x1_t1) + 1)
N1 = len(x1_t1)

# розрахунок взаємної кореляції за формулою (2)
r12_t1 = np.sum(x1_t1 * x2_t1) / N1

# розрахунок енергії за формулою (7)
E_x1_t1 = np.sum(x1_t1**2) / N1
E_x2_t1 = np.sum(x2_t1**2) / N1

print("--- Таблиця 1 ---")
print(f"r12: {r12_t1:.4f}")
print(f"E(x1): {E_x1_t1:.4f}")
print(f"E(x2): {E_x2_t1:.4f}")

# графік сигналів з таблиці 1
plt.figure(figsize=(9, 4.5))
plt.stem(n_t1, x1_t1, linefmt="b-", markerfmt="bo", basefmt="k-", label="x1(n)")
plt.stem(n_t1, x2_t1, linefmt="r--", markerfmt="rs", basefmt="k-", label="x2(n)")
plt.title("Дискретні сигнали x1(n) та x2(n) (Таблиця 1)")
plt.xlabel("Відлік n")
plt.ylabel("Амплітуда")
plt.xticks(n_t1)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 2. нормований коефіцієнт кореляції з таблиці 2
# -------------------------------------------------------------
# дані сигналів різної амплітуди
n_t2 = np.arange(0, 9)
x1_t2 = np.array([0, 3, 5, 5, 5, 2, 0.5, 0.25, 0], dtype=float)
x2_t2 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
x3_t2 = np.array([0, 9, 15, 15, 15, 6, 1.5, 0.75, 0], dtype=float)
x4_t2 = np.array([2, 2, 2, 2, 2, 0, 0, 0, 0], dtype=float)
N2 = len(x1_t2)

def calc_norm_correlation(x, y):
    # взаємна кореляція при j=0
    r_val = np.sum(x * y) / len(x)
    # нормувальний коефіцієнт за формулою (4)
    k_val = (1.0 / len(x)) * np.sqrt(np.sum(x**2) * np.sum(y**2))
    # нормована кореляція за формулою (5)
    rho_val = r_val / k_val
    return r_val, k_val, rho_val

r12_t2, k12_t2, rho12_t2 = calc_norm_correlation(x1_t2, x2_t2)
r34_t2, k34_t2, rho34_t2 = calc_norm_correlation(x3_t2, x4_t2)

E_x1_t2 = np.sum(x1_t2**2) / N2
E_x2_t2 = np.sum(x2_t2**2) / N2
E_x3_t2 = np.sum(x3_t2**2) / N2
E_x4_t2 = np.sum(x4_t2**2) / N2

print("\n--- Таблиця 2 ---")
print(f"Пара (x1, x2): r12 = {r12_t2:.4f}, K12 = {k12_t2:.4f}, rho12 = {rho12_t2:.4f}")
print(f"Пара (x3, x4): r34 = {r34_t2:.4f}, K34 = {k34_t2:.4f}, rho34 = {rho34_t2:.4f}")
print(f"E(x1): {E_x1_t2:.4f}, E(x2): {E_x2_t2:.4f}")
print(f"E(x3): {E_x3_t2:.4f}, E(x4): {E_x4_t2:.4f}")

# порівняння сигналів таблиці 2
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(n_t2, x1_t2, "o-", color="#1f77b4", label="x1(n)")
plt.step(n_t2, x2_t2, where="mid", color="#d62728", label="x2(n)")
plt.title("Пара x1(n) та x2(n)")
plt.xlabel("Відлік n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_t2, x3_t2, "s-", color="#2ca02c", label="x3(n) = 3*x1")
plt.step(n_t2, x4_t2, where="mid", color="#9467bd", label="x4(n) = 2*x2")
plt.title("Пара x3(n) та x4(n)")
plt.xlabel("Відлік n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------
# 3. кореляція та автокореляція випадкових сигналів (N=200)
# -------------------------------------------------------------
np.random.seed(42)
N_rnd = 200
X1 = np.random.randn(N_rnd)
X2 = np.random.randn(N_rnd)

max_lag = 20
lags = np.arange(max_lag + 1)

# автокореляція x1 та x2 за формулою (6)
r11 = np.zeros(len(lags))
r22 = np.zeros(len(lags))
# взаємна кореляція x1 та x2 за формулою (3)
r12 = np.zeros(len(lags))

for idx, j in enumerate(lags):
    r11[idx] = np.sum(X1[:N_rnd - j] * X1[j:]) / N_rnd
    r22[idx] = np.sum(X2[:N_rnd - j] * X2[j:]) / N_rnd
    r12[idx] = np.sum(X1[:N_rnd - j] * X2[j:]) / N_rnd

# енергії через нульову затримку автокореляції
E_X1_akf = r11[0]
E_X2_akf = r22[0]
E_X1_direct = np.sum(X1**2) / N_rnd
E_X2_direct = np.sum(X2**2) / N_rnd

print("\n--- Випадкові сигнали (N=200) ---")
print(f"E(X1) через АКФ r11(0): {E_X1_akf:.4f} (пряма сума: {E_X1_direct:.4f})")
print(f"E(X2) через АКФ r22(0): {E_X2_akf:.4f} (пряма сума: {E_X2_direct:.4f})")
print(f"АКФ r11(j) для j=0..5: {np.round(r11[:6], 4)}")
print(f"ВКФ r12(j) для j=0..5: {np.round(r12[:6], 4)}")

# графіки випадкових сигналів
plt.figure(figsize=(11, 5.5))
plt.subplot(2, 1, 1)
plt.plot(X1, color="#1f77b4", linewidth=1.2)
plt.title("Випадковий сигнал X1 (білий шум, N=200)")
plt.xlabel("Відлік n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(X2, color="#ff7f0e", linewidth=1.2)
plt.title("Випадковий сигнал X2 (білий шум, N=200)")
plt.xlabel("Відлік n")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# графіки автокореляції та взаємної кореляції
plt.figure(figsize=(11, 5.5))
plt.subplot(2, 1, 1)
plt.stem(lags, r11, linefmt="g-", markerfmt="go", basefmt="k-", label="АКФ r11(j)")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Автокореляційна функція r11(j) випадкового сигналу X1 (дельта-пік при j=0)")
plt.xlabel("Часовий зсув j")
plt.ylabel("Значення АКФ")
plt.xticks(lags)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.subplot(2, 1, 2)
plt.stem(lags, r12, linefmt="r-", markerfmt="rs", basefmt="k-", label="ВКФ r12(j)")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Функція взаємної кореляції r12(j) між двома незалежними шумами X1 та X2")
plt.xlabel("Часовий зсув j")
plt.ylabel("Значення ВКФ")
plt.xticks(lags)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
