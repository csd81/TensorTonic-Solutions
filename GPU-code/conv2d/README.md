Íme egy részletes, szakmai szempontból felépített vázlat és szerkezeti útmutató a GPU programozás beadandó dokumentációjához. Ez a struktúra közvetlenül igazodik a kiírásban szereplő **kötelező pontokhoz**, és biztosítja a minimum 4 oldalas terjedelem igényléséhez szükséges szakmai mélységet.

---

# Beadandó Dokumentáció Vázlat: 1D és 2D FIR Szűrés Numba CUDA-val

## 1. Bevezetés és Probléma-specifikáció (Célkitűzés)

* **A feladat bemutatása:** Mi a digitális jelek és képek szűrésének szerepe a modern informatikában, jelfeldolgozásban és számítógépes látásban (Computer Vision)?
* **Matematikai modell:**
* 1D diszkrét konvolúció (idősorok, audiószűrés).
* 2D konvolúció/korreláció (képek aluláteresztő Gauss-szűrése és felüláteresztő Sobel élkeresése).


* **Motiváció a GPU használatára:** Miért jelent kihívást nagyméretű képek vagy milliós elemszámú idősorok feldolgozása CPU-n (időbonyolultság: $O(N \cdot K)$ ill. $O(W \cdot H \cdot K_x \cdot K_y)$)? Hogyan segíthet a SIMT (Single Instruction, Multiple Threads) architektúra?

## 2. Irodalomkutatás és Párhuzamosítási Előzmények

* **Szakirodalmi háttér:** FIR (Finite Impulse Response) szűrők előnyei az IIR-rel szemben (garantált lineáris fázis, numerikus stabilitás).
* **Létező párhuzamosítási megközelítések:**
* CPU alapú multithreading (OpenMP, SIMD vektorizáció).
* GPU alapú megoldások (CUDA, OpenCL) és a memóriahierarchia (Global vs. Shared Memory) optimalizációs kihívásai.


* **Open-source keretrendszerek:** Hivatkozás a SciPy, NumPy, OpenCV és a Numba könyvtárak szerepére.

## 3. Az Algoritmus Szekvenciális Változata

* **Működési elv:** A CPU-alapú referencia-megvalósítás lépései.
* **Peremfeltételek kezelése:** Hogyan kezeli a rendszer a széleket (Zero-padding, Replicate/Clamp)?
* **Időbonyolultság elemzése:** Algoritmikus korlátok a szekvenciális végrehajtás során.

## 4. Párhuzamosítási Stratégiák és Teljesítménykritikus Részek

* **Teljesítménybottleneck-ek azonosítása:**
* Globális memória sávszélesség telítődése a naiv megközelítésben.
* Redundáns memóriakezelés (ugyanazokat a szomszédos pixeleket/mintákat többször olvassák ki a szálak).


* **Megoldási stratégiák GPU-n:**
* **1. Naiv globális memóriás megvalósítás:** Minden szál közvetlenül a globális memóriából olvas.
* **2. Shared Memory Tiling + Halo cells:** A szálblokkok beöltik a saját adatcsempéjüket és a peremterületeket a gyors megosztott memóriába, drasztikusan csökkentve a globális memória elérések számát.
* **3. Constant Memory:** A kis méretű szűrőmagok (kernel) elhelyezése a konstans memóriában a hardveres broadcast kihasználásához.



## 5. A GPU Változat Technikai Megvalósítása (Numba CUDA)

* **Környezet:** Python + Numba CUDA (`cuda.jit`, `@cuda.jit`).
* **Rácsszerkezet tervezése:**
* 1D rács (`blocks` és `threads_per_block`).
* 2D rács (`blockdim = (16, 16)` és dinamikus `griddim` számítás).


* **Kódrészletek és magyarázat:** A kernel-függvények legfontosabb sorainak (pl. `cuda.syncthreads()`, halo feltöltés, peremellenőrzés) dokumentálása.

## 6. Numerikus Korrektség Igazolása

* **Referencia alapú ellenőrzés:** A CPU (`SciPy` / `NumPy`) és a GPU eredmények összehasonlítása.
* **Tolerancia vizsgálat:** Az `np.allclose()` függvény alkalmazása (pl. `atol=1e-4` vagy `1e-3` pontossággal). Lebegőpontos (`float32`) kerekítési hibák elemzése.

## 7. Teljesítménymérés és Eredmények Kiértékelése

* **Mérési módszertan:**
* A JIT fordítási idő (warm-up) kiszűrése.
* Tiszta kernel-futásidő mérése hardveres `cuda.event` segítségével.
* Adatátviteli idő (Host-to-Device / Device-to-Host) vs. számítási idő vizsgálata.


* **Mérési adatok (javasolt táblázatos formában):**
* Különböző méretű bemenetek ($10^5 \text{--} 5 \cdot 10^7$ elem 1D-ben; $512 \times 512 \text{ezres}$ tól $8K$ képekig 2D-ben).
* Különböző szűrőmag-méretek ($3\times3$, $7\times7$, $15\times15$).


* **Speedup (Gyorsulási) görbék és elemzés:** Melyik bemeneti mérettől éri meg igazán a GPU használata? Hol jelenik meg a PCIe busz korlátja?

## 8. Összegzés és Következtetések

* A tapasztalatok összefoglalása.
* A naiv és az optimalizált (shared memory) GPU kernel összehasonlítása.
* További fejlesztési lehetőségek (pl. texture memory használata, többszálas stream-kezelés).

---

## 9. Mért benchmark eredmények

**Mérési környezet**

| | |
|---|---|
| GPU | NVIDIA Quadro P2000 (compute capability 6.1, sm_61) |
| CUDA driver | 580.178.04 |
| CUDA toolkit (C++) | 12.4.131 |
| Python / Numba | 3.14.7 / 0.68.1 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| Mérés dátuma | 2026-11-08 |

**Módszertan**

- GPU kernel idő: 5 bemelegítő futás után 20 `cuda.event`-tel mért futás mediánja; az adatmozgatás nincs benne.
- CPU referencia: 3 futás mediánja.
- Host↔Device idő: `time.perf_counter()` + `cuda.synchronize()`.
- Numerikus hiba: `max |CPU - GPU|`, illetve `np.allclose(atol=1e-3)`.
- Mindkét implementáció aszimmetrikus 3×3-as maggal is ellenőrzi a valódi konvolúciót; a maximális hiba 0.00e+00 lett.

**Python / Numba (`conv2d.py`), 4096×4096, 5×5 Gauss-mag**

| CPU (SciPy) | H2D | GPU kernel | D2H | Max. abszolút hiba | Gyorsulás |
|---:|---:|---:|---:|---:|---:|
| 412.56 ms | 7.56 ms | 22.58 ms | 10.02 ms | 6.10e-05 | 18.3× |

**CUDA C++ (`conv2d.cu`)**

| Kép | Mag | CPU referencia | H2D | GPU kernel | D2H | Max. hiba | Gyorsulás |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4096×4096 | 5×5 | 625.15 ms | 7.77 ms | 8.29 ms | 8.07 ms | 6.10e-05 | 75.4× |
| 2048×2048 | 3×3 | 74.32 ms | 2.78 ms | 1.16 ms | 2.07 ms | 4.58e-05 | 63.9× |
| 1024×1024 | 7×7 | 67.33 ms | 0.94 ms | 0.95 ms | 0.64 ms | 9.16e-05 | 71.0× |
| 512×512 | 15×15 | 71.51 ms | 0.48 ms | 0.85 ms | 0.21 ms | 1.98e-04 | 84.1× |

**Megjegyzések**

- A C++ CPU referencia skaláris, `double` akkumulációjú ciklus, ezért a C++ gyorsulási értékek nem hasonlíthatók közvetlenül a SciPy-hoz mért Python értékhez.
- A Python és a C++ teszt különböző véletlenszám-generátort használ (NumPy `RandomState` vs. `std::mt19937`), ezért a kimenetek nem bitre azonosak; mindkettő a saját CPU referenciájához képest 1e-3 alatt marad.
- A futási idők futásonként néhány százalékot ingadoznak; a táblázat reprezentatív mérések eredménye.
- Páros magméret nem támogatott: a `scipy.ndimage.convolve` origó-konvenciója páros méretnél eltér, ezért a kód páratlan K-t vár.
- A 10. szakaszban leírt mérési sorozat a naiv mellett a shared memory-os csempézett kernelt is tartalmazza (ld. `benchmark.py`).

---

## 10. Részletes mérési sorozat: naiv vs. shared memory-os csempézés

A `benchmark.py` a kiírásban szereplő mérési protokollt valósítja meg, és a
`conv2d.py`-ban található két kernelhez (`conv2d_global_kernel`,
`conv2d_shared_kernel`) készít összehasonlító sorozatot.

**Új fájlok**

- `conv2d.py` — új `conv2d_shared_kernel`: a blokk a saját kimeneti
  csempéjéhez tartozó bemenetet a halo cellákkal együtt egyszer tölti be a
  megosztott memóriába, így a belső ciklusban nincs globális olvasás és
  nincs határellenőrzés (a peremen kívüli elemek nullák).
- `benchmark.py` — mérési sorozat: 512², 1024², 2048², 4096² képméret × 3×3,
  5×5, 7×7, 15×15 mag, mindkét kernellel.
- `benchmark_2d.csv` — a teljes mért adatsor.
- `benchmark_2d.png` — kernel-idő és gyorsulás görbék.

**Módszertan**

- JIT bemelegítés (5 futás) minden konfigurációnál, a fordítási idő nem
  számít bele.
- A GPU kernel tiszta futásideje `cuda.event` párokkal mérve, 20 futás
  mediánja és szórása.
- A Host→Device és Device→Host másolás külön, `time.perf_counter()`-rel,
  `cuda.synchronize()` mellett, 3 futás mediánja.
- Numerikus ellenőrzés egy IEEE-754 float32 CPU referenciához
  (`@njit(fastmath=False)`), `max |CPU − GPU|` és `np.allclose(atol=1e-3)`.
- A konvolúció iránya (nem korreláció) aszimmetrikus magokkal is
  ellenőrizve van a mérési sorozat előtt; a maximális eltérés a SciPy
  referenciához k=15-re 7.3e-04 (float32 kerekítés).

**Eredmények (kernel idő, 20 futás mediánja)**

| Kép | Mag | CPU [ms] | Naiv [ms] | Shared [ms] | Shared/naiv | Max. hiba | Gyorsulás (naiv) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512² | 3×3 | 5.02 | 0.220 | 0.256 | 1.16 | 4.58e-05 | 22.8× |
| 512² | 5×5 | 10.83 | 0.455 | 0.455 | 1.00 | 4.58e-05 | 23.8× |
| 512² | 7×7 | 20.48 | 0.738 | 0.710 | 0.96 | 4.58e-05 | 27.8× |
| 512² | 15×15 | 90.89 | 2.687 | 2.402 | 0.89 | 4.58e-05 | 33.8× |
| 1024² | 3×3 | 18.55 | 0.645 | 0.755 | 1.17 | 4.58e-05 | 28.8× |
| 1024² | 5×5 | 42.99 | 1.454 | 1.454 | 1.00 | 6.10e-05 | 29.6× |
| 1024² | 7×7 | 80.84 | 2.449 | 2.337 | 0.95 | 4.58e-05 | 33.0× |
| 1024² | 15×15 | 364.82 | 10.249 | 8.780 | 0.86 | 6.10e-05 | 35.6× |
| 2048² | 3×3 | 75.83 | 2.440 | 2.897 | 1.19 | 6.10e-05 | 31.1× |
| 2048² | 5×5 | 170.91 | 5.734 | 5.724 | 1.00 | 4.58e-05 | 29.8× |
| 2048² | 7×7 | 323.36 | 10.184 | 9.768 | 0.96 | 6.10e-05 | 31.8× |
| 2048² | 15×15 | 1452.29 | 40.325 | 36.169 | 0.90 | 4.58e-05 | 36.0× |
| 4096² | 3×3 | 302.68 | 10.146 | 11.930 | 1.18 | 6.10e-05 | 29.8× |
| 4096² | 5×5 | 698.26 | 23.637 | 23.624 | 1.00 | 6.10e-05 | 29.5× |
| 4096² | 7×7 | 1297.75 | 39.976 | 38.122 | 0.95 | 6.10e-05 | 32.5× |
| 4096² | 15×15 | 5992.73 | 161.037 | 144.066 | 0.89 | 4.58e-05 | 37.2× |

**Értékelés**

- A shared memory-os csempézés csak a nagyobb magoknál ad érdemi
  nyereséget: 15×15-nél 8–14%-kal kevesebb kernel-idő, 7×7-nél ~5%-kal.
  Kis magoknál (3×3) viszont lassabb, mert a halo-többlet
  (`(16+k−1)²/16² = 1.19×` már k=3-ra) és a csempe betöltésének fix költsége
  nagyobb, mint a megspórolt globális olvasás.
- A naiv kernel a cache-hierarchiának köszönhetően eleve jól teljesít: a
  szomszédos szálak átfedő ablakai L1/L2 találatok. Mindkét változat
  lényegében egy terhelés + egy FMA per MAC mintát futtat, ezért a
  különbség kicsi; a szűk keresztmetszet nem a sávszélesség, hanem a
  futásidejű magméret miatt nem unrollozható belső ciklus
  (címzés + terhelés + FMA) és a soros akkumulátor-lánc.
- A CPU→GPU gyorsulás kernel-időre 22–42×. A teljes end-to-end időben a
  PCIe-s adatmozgatás is megjelenik (4096²-nél H2D ~9–17 ms, D2H ~10–13 ms),
  ezért ott a gyorsulás kisebb: pl. 4096²/15×15 esetén 188.0 ms (naiv),
  illetve 173.4 ms (shared) a 5992.7 ms CPU-hoz képest, azaz 31.9× / 34.6×;
  4096²/3×3-nál viszont csak 9.7× / 8.9×, mert a másolás dominál.
- További gyorsítás regiszter-blokkolással érhető el (több kimenet
  szálanként, mozgó ablak a regiszterekben); ez a C++ portban
  (`conv2d.cu`) már megvan, és ott 4096²/5×5-re 8.29 ms a kernel-idő a
  Numba 22.99 ms-ával szemben.

**Reprodukálás**

```bash
python3 benchmark.py   # -> benchmark_2d.csv, benchmark_2d.png
```
