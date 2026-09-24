import os
import time
import math
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pyttsx3

# параметри дискретизації
fs = 16000
duration = 0.4
t = np.arange(int(fs * duration)) / fs

# 1. генерація збуджуючих сигналів
# вокалізований сигнал (періодичні імпульси основного тону F0)
f0 = 130
t_period = int(fs / f0)
impulse_train = np.zeros(len(t))
impulse_train[::t_period] = 1.0

# формуємо форму голосового імпульсу (модель розенберга через низькочастотний фільтр)
b_glot = [1.0]
a_glot = [1.0, -1.9, 0.9025]
glottal_pulse = signal.lfilter(b_glot, a_glot, impulse_train)
glottal_pulse = glottal_pulse / np.max(np.abs(glottal_pulse))

# невокалізований сигнал (шум з рівномірним розподілом для фрикативних звуків)
np.random.seed(42)
noise_source = np.random.uniform(-1.0, 1.0, len(t))

# 2. реалізація цифрового резонатора формант
def get_formant_filter(f_res, bw, sampling_rate):
    # коефіцієнти iir фільтра 2-го порядку для резонансного піку
    r = np.exp(-np.pi * bw / sampling_rate)
    theta = 2.0 * np.pi * f_res / sampling_rate
    a = [1.0, -2.0 * r * np.cos(theta), r * r]
    b = [1.0 - 2.0 * r * np.cos(theta) + r * r]
    return b, a

def synthesize_vowel(f0_pitch, formants, bandwidths, duration_s=0.4, sampling_rate=16000):
    n_samples = int(sampling_rate * duration_s)
    t_vowel = np.arange(n_samples) / sampling_rate
    period = int(sampling_rate / f0_pitch)

    # періодичні імпульси
    source = np.zeros(n_samples)
    source[::period] = 1.0

    # формуєм хвилю збудження
    b_g = [1.0]
    a_g = [1.0, -1.9, 0.9025]
    source = signal.lfilter(b_g, a_g, source)

    # каскадна фільтрація формантами F1, F2, F3
    y_filtered = source
    for f_center, bw in zip(formants, bandwidths):
        b_res, a_res = get_formant_filter(f_center, bw, sampling_rate)
        y_filtered = signal.lfilter(b_res, a_res, y_filtered)

    # врахування випромінювання губ (диференціюючий фільтр)
    y_filtered = signal.lfilter([1.0, -0.96], [1.0], y_filtered)
    y_filtered = y_filtered / np.max(np.abs(y_filtered))
    return t_vowel, y_filtered

def synthesize_fricative(duration_s=0.4, sampling_rate=16000):
    n_samples = int(sampling_rate * duration_s)
    t_fric = np.arange(n_samples) / sampling_rate
    noise = np.random.uniform(-1.0, 1.0, n_samples)

    # смуговий фільтр 4-8 кгц для фрикативного звуку [с]
    sos = signal.butter(4, [4000, 7500], btype='bandpass', fs=sampling_rate, output='sos')
    y_fric = signal.sosfilt(sos, noise)
    y_fric = y_fric / np.max(np.abs(y_fric))
    return t_fric, y_fric

# синтезуємо голосні [а], [і], [у] та фрикативний [с]
formants_a = [730, 1090, 2440]
formants_i = [270, 2290, 3010]
formants_u = [300, 870, 2240]
bw_default = [60, 90, 120]

t_a, wave_a = synthesize_vowel(f0, formants_a, bw_default, duration, fs)
t_i, wave_i = synthesize_vowel(f0, formants_i, bw_default, duration, fs)
t_u, wave_u = synthesize_vowel(f0, formants_u, bw_default, duration, fs)
t_s, wave_s = synthesize_fricative(duration, fs)

# 3. компандування по мю-закону та диференційна ікм (дікм)
def mu_law_compress(signal_in, mu=255):
    # стиснення амплітуди за логарифмічним законом
    return np.sign(signal_in) * np.log(1.0 + mu * np.abs(signal_in)) / np.log(1.0 + mu)

def mu_law_expand(signal_in, mu=255):
    # зворотне розширення амплітуди
    return np.sign(signal_in) * (1.0 / mu) * ((1.0 + mu)**np.abs(signal_in) - 1.0)

def quantize(signal_in, bits=8):
    levels = 2**bits
    step = 2.0 / levels
    q_idx = np.clip(np.floor((signal_in + 1.0) / step), 0, levels - 1)
    return (q_idx + 0.5) * step - 1.0

# порівняння рівномірного квантування та мю-закону для голосного [а]
wave_unif_q = quantize(wave_a, bits=8)
snr_uniform = 10.0 * np.log10(np.sum(wave_a**2) / np.sum((wave_a - wave_unif_q)**2))

wave_comp = mu_law_compress(wave_a, mu=255)
wave_comp_q = quantize(wave_comp, bits=8)
wave_mulaw_q = mu_law_expand(wave_comp_q, mu=255)
snr_mulaw = 10.0 * np.log10(np.sum(wave_a**2) / np.sum((wave_a - wave_mulaw_q)**2))

# дікм лінійне передбачення 1-го порядку: e[n] = x[n] - a1 * x[n-1]
a1_pred = 0.95
wave_pred = np.zeros_like(wave_a)
wave_error = np.zeros_like(wave_a)
for n in range(1, len(wave_a)):
    wave_pred[n] = a1_pred * wave_a[n-1]
    wave_error[n] = wave_a[n] - wave_pred[n]

var_original = np.var(wave_a)
var_diff = np.var(wave_error)
gain_dpcm = 10.0 * np.log10(var_original / var_diff)

# вивід розрахованих показників у консоль
print("=== ПАРАМЕТРИ АКУСТИЧНОГО СИНТЕЗУ ТА ЦОІ ===")
print(f"Частота дискретизації Fs: {fs} Гц")
print(f"Частота основного тону F0: {f0} Гц (період T0 = {1000/f0:.2f} мс, {t_period} відліків)")
print(f"Форманти [а]: F1={formants_a[0]} Гц, F2={formants_a[1]} Гц, F3={formants_a[2]} Гц")
print(f"Форманти [і]: F1={formants_i[0]} Гц, F2={formants_i[1]} Гц, F3={formants_i[2]} Гц")
print(f"Форманти [у]: F1={formants_u[0]} Гц, F2={formants_u[1]} Гц, F3={formants_u[2]} Гц")
print(f"Відношення сигнал/шум (Uniform PCM 8-bit): {snr_uniform:.2f} дБ")
print(f"Відношення сигнал/шум (mu-law PCM 8-bit): {snr_mulaw:.2f} дБ")
print(f"Дисперсія вихідного сигналу [a]: {var_original:.6f}")
print(f"Дисперсія сигналу помилки ДІКМ: {var_diff:.6f}")
print(f"Зниження дисперсії (DPCM Gain): {gain_dpcm:.2f} дБ")

# 4. візуалізація інтерактивних графіків
# графік 1: джерела збудження
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.plot(t[:200] * 1000, impulse_train[:200], color="#1f77b4", linewidth=1.5)
plt.title("Імпульсна послідовність (F0 = 130 Гц)")
plt.xlabel("Час (мс)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 2, 2)
plt.plot(t[:200] * 1000, glottal_pulse[:200], color="#2ca02c", linewidth=1.5)
plt.title("Голосовий імпульс зв'язок (Rosenberg)")
plt.xlabel("Час (мс)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 2, 3)
plt.plot(t[:200] * 1000, noise_source[:200], color="#d62728", linewidth=0.8)
plt.title("Турбулентний шум (невокалізоване джерело)")
plt.xlabel("Час (мс)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

# амплітудний спектр голосового джерела vs шуму
f_axis = np.fft.rfftfreq(len(t), 1.0 / fs)
spec_glot = np.abs(np.fft.rfft(glottal_pulse))
spec_noise = np.abs(np.fft.rfft(noise_source))

plt.subplot(2, 2, 4)
plt.plot(f_axis, 20 * np.log10(spec_glot / np.max(spec_glot) + 1e-6), color="#2ca02c", label="Вокалізований")
plt.plot(f_axis, 20 * np.log10(spec_noise / np.max(spec_noise) + 1e-6), color="#d62728", alpha=0.5, label="Шум")
plt.title("Спектральна густина джерел")
plt.xlabel("Частота (Гц)")
plt.ylabel("Рівень (дБ)")
plt.legend(loc="upper right")
plt.xlim(0, 4000)
plt.ylim(-60, 5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# графік 2: амплітудно-частотні характеристики формантних фільтрів
plt.figure(figsize=(10, 4.5))
w_freqs = np.linspace(0, 4000, 1000)

for name, formants, col in [("[а]", formants_a, "#1f77b4"), ("[і]", formants_i, "#2ca02c"), ("[у]", formants_u, "#d62728")]:
    resp_total = np.ones_like(w_freqs, dtype=complex)
    for f_c, bw in zip(formants, bw_default):
        b, a = get_formant_filter(f_c, bw, fs)
        _, h = signal.freqz(b, a, worN=w_freqs, fs=fs)
        resp_total *= h
    h_db = 20 * np.log10(np.abs(resp_total) + 1e-6)
    plt.plot(w_freqs, h_db, label=f"Голосний {name} (F1={formants[0]}, F2={formants[1]}, F3={formants[2]} Гц)", color=col, linewidth=2)

plt.title("АЧХ передавальної функції голосового тракту для голосних звуків")
plt.xlabel("Частота (Гц)")
plt.ylabel("Коефіцієнт передачі (дБ)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# графік 3: часові хвилі синтезованих звуків
plt.figure(figsize=(10, 6))
plot_samples = int(fs * 0.04) # 40 мс для чіткої періодичної структури

plt.subplot(2, 2, 1)
plt.plot(t[:plot_samples] * 1000, wave_a[:plot_samples], color="#1f77b4", linewidth=1.5)
plt.title("Синтезований звук [а]")
plt.xlabel("Час (мс)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 2, 2)
plt.plot(t[:plot_samples] * 1000, wave_i[:plot_samples], color="#2ca02c", linewidth=1.5)
plt.title("Синтезований звук [і]")
plt.xlabel("Час (мс)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 2, 3)
plt.plot(t[:plot_samples] * 1000, wave_u[:plot_samples], color="#ff7f0e", linewidth=1.5)
plt.title("Синтезований звук [у]")
plt.xlabel("Час (мс)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 2, 4)
plt.plot(t[:plot_samples] * 1000, wave_s[:plot_samples], color="#d62728", linewidth=1.0)
plt.title("Фрикативний звук [с]")
plt.xlabel("Час (мс)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# графік 4: спектри голосних та спектрограма послідовності [а]-[і]-[у]-[с]
full_speech = np.concatenate([wave_a, wave_i, wave_u, wave_s])
t_full = np.arange(len(full_speech)) / fs

plt.figure(figsize=(11, 5.5))
plt.subplot(2, 1, 1)
plt.plot(t_full, full_speech, color="#333333", linewidth=1)
plt.title("Синтезована послідовність звуків [а] -> [і] -> [у] -> [с]")
plt.xlabel("Час (с)")
plt.ylabel("Амплітуда")
plt.grid(True, linestyle="--", alpha=0.6)

plt.subplot(2, 1, 2)
f_sgram, t_sgram, sxx = signal.spectrogram(full_speech, fs, nperseg=256, noverlap=128)
plt.pcolormesh(t_sgram, f_sgram, 10 * np.log10(sxx + 1e-6), shading='gouraud', cmap='magma')
plt.title("Спектрограма синтезованого вербального сигналу")
plt.xlabel("Час (с)")
plt.ylabel("Частота (Гц)")
plt.ylim(0, 5000)
plt.colorbar(label="Потужність (дБ)")
plt.tight_layout()
plt.show()

# 5. компонент голосового помічника (tts синтез)
def execute_voice_command(cmd_text, output_audio_dir="."):
    # обробка команд та синтез голосової відповіді
    cmd_text_clean = cmd_text.strip().lower()

    if "привіт" in cmd_text_clean or "hello" in cmd_text_clean:
        response_text = "Привіт! Аудіосистему та голосовий помічник активовано."
    elif "час" in cmd_text_clean or "time" in cmd_text_clean:
        cur_time = time.strftime("%H:%M:%S")
        response_text = f"Поточний системний час: {cur_time}."
    elif "стан" in cmd_text_clean or "status" in cmd_text_clean:
        response_text = "Усі підсистеми ЦОІ та цифровий синтезатор формант працюють у штатному режимі."
    elif "обчисли" in cmd_text_clean or "calculate" in cmd_text_clean:
        response_text = "Результат обчислення формантних коефіцієнтів рівний ста відсоткам."
    elif "бувай" in cmd_text_clean or "exit" in cmd_text_clean or "побачення" in cmd_text_clean:
        response_text = "Сеанс роботи голосового помічника завершено. До побачення!"
    else:
        response_text = f"Команду '{cmd_text}' прийнято до обробки синтезатором мови."

    print(f"\n[Користувач]: {cmd_text}")
    print(f"[Голосовий помічник]: {response_text}")

    # збереження аудіофайлу через pyttsx3
    try:
        engine = pyttsx3.init()
        audio_file = os.path.join(output_audio_dir, f"speech_response_{int(time.time())}.wav")
        engine.save_to_file(response_text, audio_file)
        engine.runAndWait()
        print(f"Збережено аудіофайл відповіді: {audio_file}")
    except Exception as e:
        print(f"Помилка синтезу аудіо: {e}")

    return response_text

print("\n=== ТЕСТУВАННЯ МОДУЛЯ ГОЛОСОВОГО ПОМІЧНИКА ===")
test_commands = ["Hello assistant", "Поточний час", "Стан системи", "До побачення"]
for cmd in test_commands:
    execute_voice_command(cmd, output_audio_dir=".")
