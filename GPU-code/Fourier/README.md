A Short-Time Fourier Transform (STFT) GPU-s megvalósítása Python + Numba környezetben kiváló témaválasztás: jól illeszkedik a kötelező követelményekhez, szigorúan demonstrálja a memóriaarchitektúra (Shared Memory, coalesced elérés) kezelését, és természetes összehasonlítási alapot nyújt a CPU-s könyvtárakkal (pl. `scipy.signal.stft`, NumPy) szemben.

---

### 1. Elméleti alapok és a feladat matematikai specifikációja

Az STFT egy folytonos vagy diszkrét idejű jelet kisebb, átfedő ablakokra bont, és ezeken külön-külön hajt végre diszkrét Fourier-transzformációt (DFT).

Egy $x[n]$ bemeneti diszkrét jelre, $w[m]$ ablakfüggvényre ($M$ hosszúság) és $H$ lépésközre (hop size / stride) a $k$-adik időablak $m$-edik frekvenciakomponense:

$$X[k, m] = \sum_{n=0}^{M-1} x[k \cdot H + n] \cdot w[n] \cdot e^{-j \frac{2\pi}{N} m n}, \quad m = 0, 1, \dots, N-1$$

ahol:

* $M$: ablak hossza (pl. 256, 512, 1024 minta),
* $H$: átfedési lépésköz ($H < M$, gyakran $M/2$ vagy $M/4$),
* $N$: az FFT pontszáma ($N \ge M$, szükség esetén zérókitöltéssel),
* $k$: az időbeli keret (frame) indexe ($0 \le k < K$).

---

### 2. Szekvenciális feldolgozás lépései (CPU referencia)

A szekvenciális CPU folyamat lépései:

1. **Ablakolás (Framing & Windowing):** Adott $k$ indexű kerethez a jelszakasz kimásolása: $s_k[n] = x[k \cdot H + n] \cdot w[n]$.
2. **Padding:** Ha $N > M$, $s_k$ kiegészítése nullákkal.
3. **1D DFT / FFT számítás:** A transzformáció elvégzése keretenként: $X[k] = \text{FFT}(s_k)$.
4. **Magnitúdó / Teljesítményspektrum (opcionális):** $\vert{}X[k, m]\vert{}$ vagy $\vert{}X[k, m]\vert{}^2$ képzése a spektrogramhoz.

---

### 3. Párhuzamosítási stratégiák és GPU kihívások Numbával

A feladatfelhívás rögzíti, hogy a teljes algoritmust nem helyettesítheti kész könyvtár (pl. nem fogadható el egy sima `cupy.fft` hívás). Numbában a GPU párhuzamosítás két fő úton építhető fel:

#### A stratégia: Párhuzamos batched DFT kernel (Oktatási szempontból a legtisztább)

Ha a keretméret mérsékelt ($M, N \le 1024$), a keretenkénti DFT közvetlenül kiértékelhető GPU szálakkal.

* **Hierarchia:** Minden CUDA Thread Block egy időbeli keretet ($k$) dolgoz fel.
* **Szálak:** A blokkon belüli $t$ indexű szál egy vagy több frekvenciakomponens ($m$) összegzését számítja ki.
* **Memóriahierarchia optimalizálás:**
* Az ablakfüggvény $w[n]$ betölthető **Constant Memory**-ba vagy **Shared Memory**-ba.
* Az aktuális keret $x[k \cdot H + n]$ adatai a globális memóriából egyszerre, **coalesced** olvasással bekerülnek a blokk **Shared Memory**-jába (`cuda.shared.array`). Ezzel elkerülhető, hogy minden szál újra a globális memóriából olvassa a mintákat a szorzásokhoz.



#### B stratégia: Fúzionált Ablakolás + Batched Radix-2 FFT kernel

Nagyobb pontszámok esetén Numbában implementálható egy egyszerű Radix-2 Cooley-Tukey FFT kernel, ahol:

* 1. fázis: Bemeneti minta beolvasása Shared Memory-ba + ablakozás ($x \cdot w$) + bit-reversal permutáció.


* 2. fázis: $\log_2(N)$ lépéses butterfly hálózat végrehajtása `cuda.syncthreads()` szinkronizációval.



---

### 4. GPU Implementációs architektúra (Numba CUDA mintavázlat)

Egy tipikus, shared memory-t használó batched STFT kernel vázlata Numbában:

```python
import math
import numpy as np
from numba import cuda

@cuda.jit
def stft_dft_kernel(d_signal, d_window, d_real_out, d_imag_out, 
                    signal_len, frame_len, hop_size, n_frames, n_fft):
    """
    1 Thread Block = 1 Időkeret (Frame)
    Thread index (tx) = 1 Frekvenciapont (Bin)
    """
    # Shared memory a keret jelszeletének gyors eléréséhez
    s_frame = cuda.shared.array(shape=1024, dtype=np.float32)

    frame_idx = cuda.blockIdx.x
    tx = cuda.threadIdx.x

    if frame_idx >= n_frames:
        return

    frame_start = frame_idx * hop_size

    # 1. Adatbetöltés és ablakozás Shared Memory-ba (coalesced olvasás)
    i = tx
    while i < n_fft:
        if i < frame_len and (frame_start + i) < signal_len:
            s_frame[i] = d_signal[frame_start + i] * d_window[i]
        else:
            s_frame[i] = 0.0  # Zero-padding
        i += cuda.blockDim.x

    cuda.syncthreads()

    # 2. DFT összegzés szála(n)ként (tx = frekvenciabin index)
    m = tx
    while m < (n_fft // 2 + 1):  # RFFT: csak a pozitív frekvenciák kellenek
        sum_real = 0.0
        sum_imag = 0.0
        angle_step = -2.0 * math.pi * m / n_fft

        for n in range(n_fft):
            angle = angle_step * n
            val = s_frame[n]
            sum_real += val * math.cos(angle)
            sum_imag += val * math.sin(angle)

        d_real_out[frame_idx, m] = sum_real
        d_imag_out[frame_idx, m] = sum_imag

        m += cuda.blockDim.x

```

---

### 5. Dokumentáció és mérési struktúra tervezete

| Fejezet | Tartalmi fókusz a beadandóban |
| --- | --- |
| **1. Problémaspecifikáció és Irodalom** | Idő-frekvencia felbontás (Heisenberg-Gábor bizonytalansági reláció). Alkalmazási területek: rezgésdiagnosztika, audio/beszédfeldolgozás, radar/szonár. Létező GPU megoldások (cuFFT batched 1D, torchaudio STFT). |
| **2. Szekvenciális algoritmus** | Naiv NumPy szeletelés és DFT/FFT implementáció ismertetése pszeudokóddal és műveletigény-számítással ($\mathcal{O}(K \cdot N^2)$ vagy $\mathcal{O}(K \cdot N \log N)$). |
| **3. Párhuzamosítási lehetőségek** | Profilozás elemzése (a CPU-n hol van a szűk keresztmetszet), párhuzamos dekompozíció (frame-szintű és bin-szintű párhuzamosság), memória-sávszélesség korlátok és bank conflict kezelése. |
| **4. GPU változat bemutatása** | Grid/Block dimenziók meghatározása, Numba CUDA kernel felépítése, `cuda.shared.array` használat indoklása a globális memória latency elfedésére, aszinkron adatátvitel (Host-Device memória transzferek). |
| **5. Numerikus verifikáció** | Összehasonlítás a `scipy.signal.stft` vagy `numpy.fft.rfft` kimenetével. Eltérések metrikái: maximális abszolút hiba ($\max \vert{}X_{\text{ref}} - X_{\text{gpu}}\vert{}$) és relatív $L_2$ norma (Frobenius-norma) $10^{-5}$ körüli toleranciával float32 esetén. |
| **6. Teljesítménymérés és kiértékelés** | Bemeneti jelhossz ($10^5 \dots 10^7$ minta) és ablakméret ($M \in \{256, 512, 1024\}$) skálázása. Speedup görbék (CPU szekvenciális vs. Numba CPU vs. Numba GPU), memóriamásolási idő különválasztása a tiszta kernelidőtől. |