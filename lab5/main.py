import numpy as np
import scipy.fftpack as fftpack
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import io

# генерація базового тестового контейнера 256x256
def create_host_image():
    x = np.linspace(-3, 3, 256)
    y = np.linspace(-3, 3, 256)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    base = 130 + 55 * np.sin(xx * 1.6) * np.cos(yy * 1.6) + 35 * np.exp(-r**2 / 3.5)
    base[:64, :64] += 25
    base[192:, 192:] -= 25
    return np.clip(base, 0, 255).astype(np.uint8)

# генерація бінарного водяного знаку 32x32
def create_watermark():
    img = Image.new('L', (32, 32), color=0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, 27, 27], outline=255, width=2)
    draw.line([4, 4, 27, 27], fill=255, width=2)
    draw.line([4, 27, 27, 4], fill=255, width=2)
    draw.rectangle([11, 11, 20, 20], fill=255)
    return (np.array(img) > 128).astype(np.uint8)

# видимий оптичний маркер через альфа-змішування
def embed_visible(host, logo, alpha=0.35):
    res = host.copy().astype(np.float64)
    h, w = host.shape
    lh, lw = logo.shape
    # масштабуєм логотип під кутовий блок
    logo_resized = np.array(Image.fromarray((logo * 255).astype(np.uint8)).resize((64, 64)))
    rh, rw = logo_resized.shape
    y0, x0 = h - rh - 12, w - rw - 12
    res[y0:y0+rh, x0:x0+rw] = (1 - alpha) * res[y0:y0+rh, x0:x0+rw] + alpha * logo_resized
    return np.clip(np.round(res), 0, 255).astype(np.uint8)

# lsb вбудовування
def embed_lsb(host, wm):
    stego = host.copy()
    wh, ww = wm.shape
    # ховаєм у молодший біт верхньо-лівого блоку або розтираєм
    stego[:wh, :ww] = (stego[:wh, :ww] & 0xFE) | wm
    return stego

# lsb вилучення
def extract_lsb(stego, shape=(32, 32)):
    wh, ww = shape
    return stego[:wh, :ww] & 0x01

# 2d dct та idct для блоків
def dct2(b):
    return fftpack.dct(fftpack.dct(b.T, norm='ortho').T, norm='ortho')

def idct2(b):
    return fftpack.idct(fftpack.idct(b.T, norm='ortho').T, norm='ortho')

# dct вбудовування в середні частоти блоків 8x8
def embed_dct(host, wm, delta=38.0, u1=2, v1=3, u2=3, v2=2):
    stego = host.astype(np.float64).copy()
    wh, ww = wm.shape
    for i in range(wh):
        for j in range(ww):
            block = stego[i*8:(i+1)*8, j*8:(j+1)*8]
            dct_b = dct2(block)
            bit = wm[i, j]
            c1, c2 = dct_b[u1, v1], dct_b[u2, v2]
            if bit == 1:
                if c1 - c2 < delta:
                    avg = (c1 + c2) / 2.0
                    dct_b[u1, v1] = avg + delta / 2.0
                    dct_b[u2, v2] = avg - delta / 2.0
            else:
                if c2 - c1 < delta:
                    avg = (c1 + c2) / 2.0
                    dct_b[u1, v1] = avg - delta / 2.0
                    dct_b[u2, v2] = avg + delta / 2.0
            stego[i*8:(i+1)*8, j*8:(j+1)*8] = idct2(dct_b)
    return np.clip(np.round(stego), 0, 255).astype(np.uint8)

# dct сліпе вилучення
def extract_dct(stego, shape=(32, 32), u1=2, v1=3, u2=3, v2=2):
    wh, ww = shape
    ext = np.zeros((wh, ww), dtype=np.uint8)
    for i in range(wh):
        for j in range(ww):
            block = stego[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float64)
            dct_b = dct2(block)
            ext[i, j] = 1 if dct_b[u1, v1] > dct_b[u2, v2] else 0
    return ext

# обчислення mse та psnr
def calc_metrics(orig, marked):
    mse = np.mean((orig.astype(np.float64) - marked.astype(np.float64))**2)
    if mse == 0:
        return 0.0, float('inf')
    psnr = 10.0 * np.log10((255.0**2) / mse)
    return mse, psnr

# коефіцієнт помилкових бітів ber (%)
def calc_ber(orig_wm, ext_wm):
    return float(np.mean(orig_wm != ext_wm) * 100.0)

# нормована крос-кореляція ncc
def calc_ncc(orig_wm, ext_wm):
    w1 = orig_wm.astype(np.float64)
    w2 = ext_wm.astype(np.float64)
    denom = np.sqrt(np.sum(w1**2) * np.sum(w2**2))
    if denom == 0:
        return 0.0
    return float(np.sum(w1 * w2) / denom)

# моделювання атак на контейнер
def attack_gaussian_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

def attack_jpeg(img, quality=50):
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))

def attack_brightness(img, offset=30):
    return np.clip(img.astype(np.float64) + offset, 0, 255).astype(np.uint8)

def attack_blur(img):
    pil_img = Image.fromarray(img)
    blurred = pil_img.filter(ImageFilter.BoxBlur(1))
    return np.array(blurred)

def main():
    np.random.seed(42)
    host = create_host_image()
    wm = create_watermark()

    # 1. видимий оптичний маркер
    visible_stego = embed_visible(host, wm, alpha=0.4)
    vis_mse, vis_psnr = calc_metrics(host, visible_stego)

    # 2. просторове lsb маркування
    lsb_stego = embed_lsb(host, wm)
    lsb_ext = extract_lsb(lsb_stego, wm.shape)
    lsb_mse, lsb_psnr = calc_metrics(host, lsb_stego)
    lsb_ber = calc_ber(wm, lsb_ext)
    lsb_ncc = calc_ncc(wm, lsb_ext)

    # 3. частотне dct маркування
    dct_stego = embed_dct(host, wm, delta=38.0)
    dct_ext = extract_dct(dct_stego, wm.shape)
    dct_mse, dct_psnr = calc_metrics(host, dct_stego)
    dct_ber = calc_ber(wm, dct_ext)
    dct_ncc = calc_ncc(wm, dct_ext)

    print("=== ПОКАЗНИКИ ЯКОСТІ МАРКУВАННЯ (БЕЗ АТАК) ===")
    print(f"Видимий маркер:  MSE = {vis_mse:.4f}, PSNR = {vis_psnr:.2f} dB")
    print(f"LSB маркер:      MSE = {lsb_mse:.4f}, PSNR = {lsb_psnr:.2f} dB, BER = {lsb_ber:.2f}%, NCC = {lsb_ncc:.4f}")
    print(f"DCT маркер:      MSE = {dct_mse:.4f}, PSNR = {dct_psnr:.2f} dB, BER = {dct_ber:.2f}%, NCC = {dct_ncc:.4f}")
    print()

    # дослідження стійкості до атак
    attacks = {
        "Gaussian Noise (sigma=15)": (attack_gaussian_noise(lsb_stego, 15), attack_gaussian_noise(dct_stego, 15)),
        "JPEG Compression (Q=50)":   (attack_jpeg(lsb_stego, 50), attack_jpeg(dct_stego, 50)),
        "Brightness Shift (+30)":    (attack_brightness(lsb_stego, 30), attack_brightness(dct_stego, 30)),
        "Box Blur (3x3)":            (attack_blur(lsb_stego), attack_blur(dct_stego))
    }

    print("=== ПОРІВНЯННЯ СТІЙКОСТІ LSB ТА DCT ДО АТАК ===")
    attack_results = {}
    for name, (att_lsb, att_dct) in attacks.items():
        ext_l = extract_lsb(att_lsb, wm.shape)
        ext_d = extract_dct(att_dct, wm.shape)
        ber_l = calc_ber(wm, ext_l)
        ber_d = calc_ber(wm, ext_d)
        ncc_l = calc_ncc(wm, ext_l)
        ncc_d = calc_ncc(wm, ext_d)
        _, psnr_l = calc_metrics(host, att_lsb)
        _, psnr_d = calc_metrics(host, att_dct)
        attack_results[name] = (att_lsb, att_dct, ext_l, ext_d, ber_l, ber_d, ncc_l, ncc_d, psnr_l, psnr_d)
        print(f"Атака: {name}")
        print(f"  LSB: BER = {ber_l:6.2f}%, NCC = {ncc_l:.4f}, Container PSNR = {psnr_l:.2f} dB")
        print(f"  DCT: BER = {ber_d:6.2f}%, NCC = {ncc_d:.4f}, Container PSNR = {psnr_d:.2f} dB")
    print()

    # ---------------------------------------------------------
    # графік 1: видимий маркер та різницева карта
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 3, 1)
    plt.imshow(host, cmap='gray', vmin=0, vmax=255)
    plt.title("Оригінальний контейнер (256x256)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(visible_stego, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Видимий ОМ (alpha=0.4)\nPSNR = {vis_psnr:.2f} dB")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    diff_vis = np.abs(host.astype(float) - visible_stego.astype(float))
    plt.imshow(diff_vis, cmap='hot')
    plt.title("Різницева карта |I - I_marked|")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # графік 2: просторовий метод lsb
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 4.5))
    plt.subplot(1, 4, 1)
    plt.imshow(wm, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Водяний знак (32x32)")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(lsb_stego, cmap='gray', vmin=0, vmax=255)
    plt.title(f"LSB контейнер\nPSNR = {lsb_psnr:.2f} dB")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    diff_lsb = np.abs(host.astype(float) - lsb_stego.astype(float)) * 200.0
    plt.imshow(diff_lsb, cmap='inferno')
    plt.title("LSB різниця (x200)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(lsb_ext, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Вилучений LSB знак\nBER = {lsb_ber:.1f}%, NCC = {lsb_ncc:.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # графік 3: частотний метод dct
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 4.5))
    plt.subplot(1, 4, 1)
    plt.imshow(host, cmap='gray', vmin=0, vmax=255)
    plt.title("Оригінальний контейнер")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(dct_stego, cmap='gray', vmin=0, vmax=255)
    plt.title(f"DCT контейнер (delta=38)\nPSNR = {dct_psnr:.2f} dB")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    diff_dct = np.abs(host.astype(float) - dct_stego.astype(float)) * 10.0
    plt.imshow(diff_dct, cmap='magma')
    plt.title("DCT різниця (x10)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(dct_ext, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Вилучений DCT знак\nBER = {dct_ber:.1f}%, NCC = {dct_ncc:.2f}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # графік 4: матриця порівняння стійкості до атак
    # ---------------------------------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(13, 11))
    row = 0
    for name, data in attack_results.items():
        att_lsb, att_dct, ext_l, ext_d, ber_l, ber_d, ncc_l, ncc_d, psnr_l, psnr_d = data

        axes[row, 0].imshow(att_lsb, cmap='gray')
        axes[row, 0].set_title(f"{name}\nLSB cont. (PSNR={psnr_l:.1f}dB)", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(ext_l, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(f"LSB вилучено\nBER={ber_l:.1f}%, NCC={ncc_l:.2f}", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(att_dct, cmap='gray')
        axes[row, 2].set_title(f"{name}\nDCT cont. (PSNR={psnr_d:.1f}dB)", fontsize=9)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(ext_d, cmap='gray', vmin=0, vmax=1)
        axes[row, 3].set_title(f"DCT вилучено\nBER={ber_d:.1f}%, NCC={ncc_d:.2f}", fontsize=9)
        axes[row, 3].axis("off")
        row += 1

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # графік 5: кількісні залежності ber та psnr
    # ---------------------------------------------------------
    # залежність ber від якості jpeg стиснення
    q_vals = np.arange(10, 95, 10)
    ber_lsb_q = []
    ber_dct_q = []
    for q in q_vals:
        att_l = attack_jpeg(lsb_stego, quality=int(q))
        att_d = attack_jpeg(dct_stego, quality=int(q))
        ber_lsb_q.append(calc_ber(wm, extract_lsb(att_l, wm.shape)))
        ber_dct_q.append(calc_ber(wm, extract_dct(att_d, wm.shape)))

    # залежність ber від потужності гаусового шуму
    sig_vals = np.arange(2, 32, 4)
    ber_lsb_s = []
    ber_dct_s = []
    for s in sig_vals:
        att_l = attack_gaussian_noise(lsb_stego, sigma=s)
        att_d = attack_gaussian_noise(dct_stego, sigma=s)
        ber_lsb_s.append(calc_ber(wm, extract_lsb(att_l, wm.shape)))
        ber_dct_s.append(calc_ber(wm, extract_dct(att_d, wm.shape)))

    # залежність psnr від сили вбудовування delta у dct
    deltas = np.linspace(10, 80, 15)
    psnr_deltas = []
    for d in deltas:
        s_d = embed_dct(host, wm, delta=d)
        _, p_d = calc_metrics(host, s_d)
        psnr_deltas.append(p_d)

    plt.figure(figsize=(14, 4.2))

    plt.subplot(1, 3, 1)
    plt.plot(q_vals, ber_lsb_q, 'r-o', label='LSB (просторовий)', linewidth=2)
    plt.plot(q_vals, ber_dct_q, 'g-s', label='DCT (частотний)', linewidth=2)
    plt.title("Стійкість до JPEG стиснення")
    plt.xlabel("Параметр якості JPEG (Quality)")
    plt.ylabel("Помилка BER (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(sig_vals, ber_lsb_s, 'r-o', label='LSB (просторовий)', linewidth=2)
    plt.plot(sig_vals, ber_dct_s, 'g-s', label='DCT (частотний)', linewidth=2)
    plt.title("Стійкість до Гаусового шуму")
    plt.xlabel("СКП шуму sigma")
    plt.ylabel("Помилка BER (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(deltas, psnr_deltas, 'b-^', linewidth=2)
    plt.axhline(30, color='red', linestyle=':', label='Поріг непомітності (30 dB)')
    plt.title("Якість контейнера від сили DCT (Delta)")
    plt.xlabel("Сила модуляції коефіцієнтів Delta")
    plt.ylabel("PSNR контейнера (dB)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
