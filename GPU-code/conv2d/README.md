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
- A 2D implementáció jelenleg csak a naiv globális memóriás kernelt tartalmazza; a shared memory-os csempézés a következő optimalizálási lépés.
