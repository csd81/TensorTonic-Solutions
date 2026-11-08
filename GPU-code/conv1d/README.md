Az **1D és 2D konvolúció/korreláció (FIR szűrés) Numba CUDA-val** kiváló választás a beadandóhoz: a feladat matematikailag jól definiált, rendkívül látványos gyorsulást produkál GPU-n, és lehetővé teszi több különböző GPU-memóriakezelési technika fokozatos bemutatását.

---

### A feladat elméleti háttere és specifikációja

A diszkrét konvolúció és keresztkorreláció a digitális jelfeldolgozás (DSP) és képfeldolgozás alapművelete:

* **1D FIR szűrés (időbeli jelek):**

$$y[n] = \sum_{k=0}^{K-1} x[n - k] \cdot h[k]$$



Ahol $x$ a bemeneti diszkrét jel, $h$ a $K$ hosszúságú véges impulzusválaszú (FIR) szűrőmag (kernel), $y$ pedig a szűrt kimenet.
* **2D FIR szűrés (képek, mátrixok):**

$$Y[i, j] = \sum_{m=-M}^{M} \sum_{n=-N}^{N} X[i - m, j - n] \cdot H[m, n]$$



Képeknél az aluláteresztő szűrés (pl. Gauss-szűrő, átlagoló maszk) zajcsökkentésre, a felüláteresztő (pl. Laplace- vagy Sobel-maszk) élkeresésre és kontrasztkiemelésre szolgál.
* **Konvolúció vs. Korreláció:** A konvolúció során a szűrőmag indexelése megfordul ($h[-k]$), míg a keresztkorrelációnál azonos irányú. Szimmetrikus szűrőmagok (mint a tipikus Gauss- vagy átlagoló szűrők) esetén a kettő numerikusan megegyezik.

---

### Párhuzamosítási stratégiák Numba CUDA környezetben

A jelfeldolgozási feladat jellege "embarrassingly parallel", hiszen a kimeneti vektor vagy mátrix minden eleme függetlenül kiszámítható. A GPU-n az alábbi három megközelítés lépcsőzetes vizsgálata adja a legerősebb mérési anyagot:

1. **Globális memóriás (Naive) megvalósítás:**
* Minden szál (thread) egy kimeneti pontot számol ki.
* A szűrőmag és a bemeneti adatok közvetlenül a lassabb globális memóriából (`cuda.to_device`) kerülnek kiolvasásra.
* Szomszédos szálak ugyanazokat a bemeneti pontokat olvassák újra, ami memóriasávszélesség-pazarló.


2. **Állandó memóriás (Constant Memory) szűrőmag:**
* Mivel a szűrőmag ($h$ vagy $H$) mérete kicsi ($3 \times 3$, $7 \times 7$, vagy 1D-ben $31\text{--}127$ pont), és minden szál ugyanazokat az együtthatókat olvassa egyszerre, érdemes a dedikált konstans memóriába (`cuda.const.array_like`) tölteni. Ez hardveres broadcast gyorsítótárat használ.


3. **Megosztott memóriás csempézés (Shared Memory Tiling + Halo):**
* A szálblokkok egy közös csempét (tile) töltenek be a blokkhoz tartozó gyors `cuda.shared.array`-be.
* A szűréshez szükséges peremterületet (halo/apron cellák) a blokk szélső szálai töltik be. Ezzel drasztikusan csökkenthető a globális memória elérések száma.



---

### Javasolt architektúra és mérések a dokumentációhoz

| Mérési szempont | 1D Konvolúció (Audió/Idősor) | 2D Konvolúció (Képek/Mátrixok) |
| --- | --- | --- |
| **Bemeneti méretek** | $10^5$, $10^6$, $10^7$, $5 \times 10^7$ pont | $512 \times 512$, $2048 \times 2048$, $8192 \times 8192$ |
| **Magméretek (Kernel)** | 15, 63, 127 tap | $3 \times 3$, $7 \times 7$, $15 \times 15$ |
| **Referencia (CPU)** | `scipy.signal.convolve`, `numpy.convolve` | `scipy.ndimage.convolve`, OpenCV |
| **Numerikus ellenőrzés** | `np.allclose(cpu_out, gpu_out, atol=1e-5)` | `np.allclose(cpu_out, gpu_out, atol=1e-5)` |
| **Időmérés** | `cuda.Event` (GPU kernelfutás kizárólagos mérésére) és Host-to-Device másolási overhead külön | `cuda.Event` + teljes end-to-end időmérés (`time.perf_counter`) |

---

### A dokumentáció felépítése (4+ oldalhoz)

* **1. Bevezetés és Irodalomkutatás:** Digitális szűrők szerepe, FIR vs. IIR összehasonlítás stabilitási szempontból, GPU párhuzamosítás története a jelfeldolgozásban.
* **2. Szekvenciális algoritmus:** Ciklusok bemutatása pszeudokóddal, $O(N \cdot K)$ és $O(N^2 \cdot K^2)$ időbonyolultság elemzése.
* **3. GPU párhuzamosítási lehetőségek és kihívások:** Memóriahierarchia (Global vs Shared vs Constant), peremfeltételek kezelése (Zero-padding, Clamp to edge), szál-divergencia kérdése a széleken.
* **4. Implementáció Numba CUDA-ban:** Rácsméretezés (`griddim`, `blockdim`), 1D és 2D kernelek felépítése, konstans és megosztott memóriás megoldás.
* **5. Numerikus korrektség:** Lebegőpontos számítások (`float32` vs. `float64`) kerekítési eltéréseinek bemutatása CPU (`scipy`) és GPU között.
* **6. Teljesítménymérések és diagramok:** Speedup görbék (CPU vs. Naive GPU vs. Shared Memory GPU), adatátviteli idő (PCIe overhead) vs. számítási idő vizsgálata méret függvényében.

---

## Mért benchmark eredmények

**Mérési környezet:** NVIDIA Quadro P2000 (sm_61), driver 580.178.04, CUDA 12.4.131, Python 3.14.7, Numba 0.68.1, NumPy 2.5.4, SciPy 1.18.1; mérés: 2026-11-08.

**Módszertan**

- GPU kernel idő: 5 bemelegítő futás után 20 `cuda.event`-es futás mediánja (adatmozgatás nélkül).
- CPU referencia: 3 futás mediánja (Python: `np.convolve`; C++: skaláris `double` ciklus).
- Host↔Device: `time.perf_counter()` + `cuda.synchronize()`.
- Bemenet: N = 5 000 000, K = 31, float32; hiba: `max |CPU - GPU|`.

**Python / Numba (`conv1d.py`)**

| CPU (NumPy) | H2D | Naiv GPU | Shared GPU | D2H (naiv/shared) | Max. hiba | Gyorsulás (naiv/shared) |
|---:|---:|---:|---:|---:|---:|---:|
| 134.96 ms | 4.78 ms | 5.51 ms | 4.68 ms | 3.36 / 3.31 ms | 2.60e-07 | 24.5× / 28.8× |

**CUDA C++ (`conv1d.cu`)**

| CPU referencia | H2D | Naiv GPU | Shared GPU | D2H (naiv/shared) | Max. hiba | Gyorsulás (naiv/shared) |
|---:|---:|---:|---:|---:|---:|---:|
| 133.69 ms | 3.57 ms | 2.45 ms | 1.52 ms | 2.67 / 2.31 ms | 2.38e-07 | 54.6× / 88.2× |

**Megjegyzések**

- A C++ CPU referencia skaláris `double` ciklus, ezért a C++ gyorsulás nem hasonlítható közvetlenül a NumPy-os Python értékhez.
- A shared memory-os kernel a nagyobb magméreteknél nagyobb előnyt ad; K=31-nél a naivhoz képest mintegy 1,6×.
- A futási idők futásonként néhány százalékot ingadoznak.
