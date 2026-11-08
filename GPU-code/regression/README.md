Az egyváltozós lineáris regresszió (legkisebb négyzetek módszere) Python és **Numba** (`@cuda.jit`) alapú megvalósításához és a beadandó felépítéséhez az alábbi áttekintés és szerkezeti útmutató nyújt segítséget a követelmények maradéktalan teljesítéséhez.

---

### Matematikai háttér és szekvenciális lépések

Egyváltozós esetben az illesztendő egyenes egyenlete: $y = \beta_1 x + \beta_0$.

Adott $N$ darab $(x_i, y_i)$ adatpont esetén a legkisebb négyzetek becslése analitikusan a következő összegekre (redukciókra) vezethető vissza:

$$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i, \quad \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$$

$$\beta_1 = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2} = \frac{N \sum x_i y_i - (\sum x_i)(\sum y_i)}{N \sum x_i^2 - (\sum x_i)^2}$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

**A szekvenciális algoritmus lépései:**

1. A bemeneti $x$ és $y$ tömbök beolvasása vagy generálása.
2. Öt alapvető skalár érték kiszámítása egyetlen végigpásztázással (loop):
* $\sum x_i$
* $\sum y_i$
* $\sum x_i^2$
* $\sum x_i y_i$
* pontok száma ($N$)


3. A meredekség ($\beta_1$) és a tengelymetszet ($\beta_0$) kiszámítása egyszerű skalár aritmetikával.

---

### Párhuzamosítási stratégiák és GPU kihívások

A feladat teljesítménykritikus része a **több millió/milliárd adatponton végzett párhuzamos összegzés (parallel reduction)**.

* **Fő kihívás:** Skalár szorzás és elemi műveletek teljesen függetlenek (embarrassingly parallel), viszont a végső összeadás szinkronizációt és szálak közötti kommunikációt igényel.
* **Lehetséges stratégiák:**
* **Megosztott memória (Shared Memory) és blokkszintű redukció:** A szálak lokálisan, blokkonként összegzik az értékeket a gyors megosztott memóriában, minimalizálva a globális memóriához való hozzáférést.
* **Atomi műveletek (`cuda.atomic.add`):** Minden szál vagy blokk közvetlenül a globális összegző változókhoz adja hozzá a részösszegeket. Nagy elemszám esetén a közvetlen globális atomi írás memória-szűk keresztmetszetet (contentions) okozhat, ezért célszerűbb blokkonként aggregálni, és csak a blokkeredményeket atomi módon összegezni.
* **Többmenetes (multi-pass) redukció:** Egy kernel kiszámítja a blokkszintű részösszegeket egy köztes tömbbe, majd egy következő kernel lépés redukálja ezt egyetlen értékké.



---

### Tervezett GPU architektúra Numbában

A Numba CUDA kernel megvalósításakor a tipikus felépítés:

1. **Host-to-Device másolás:** Adatátvitel a GPU memóriájába (`cuda.to_device`).
2. **GPU Kernel:**
* Minden szál kiszámítja egy vagy több elemhez a lokális szorzatokat ($x_i, y_i, x_i^2, x_i y_i$).
* A szálak blokkon belül megosztott memórián (`cuda.shared.array`) keresztül bináris fa struktúrában redukálják a 4 részösszeget.
* A blokk első szála (`tid == 0`) `cuda.atomic.add` hívással hozzáadja a blokk eredményét a globális kimeneti tömbhöz.


3. **Device-to-Host visszamásolás és befejezés:** A 4 skalár érték visszamásolása a CPU-ra, ahol $\mathcal{O}(1)$ idő alatt előáll a végleges $\beta_0$ és $\beta_1$.

---

### A dokumentáció javasolt felépítése (min. 4 oldal)

* **1. Probléma specifikációja és szakirodalmi áttekintés:**
* A legkisebb négyzetek módszerének elmélete, statisztikai és gépi tanulási alkalmazási területei (trendvonal-illesztés, kalibráció, előrejelzés nagy adatbázisokon).
* GPU-s redukciós technikák irodalma (pl. Mark Harris: *Optimizing Parallel Reduction in CUDA*).


* **2. Szekvenciális algoritmus:**
* Pszeudokód, lépések idő- és tárbonyolultsága ($\mathcal{O}(N)$ idő, $\mathcal{O}(1)$ tárhely).


* **3. Párhuzamosítási stratégiák és szűk keresztmetszetek:**
* Memóriasávszélesség vs. számítási kapacitás (memory-bound jelleg bemutatása).
* Redukciós minták, warp divergence és shared memory bank conflict elkerülése.


* **4. GPU változat bemutatása (Numba CUDA):**
* Grid- és blokkméretezés (pl. 256 vagy 512 szál/blokk).
* Kernel struktúra, memóriaallokáció, adatmozgatás minimalizálása.


* **5. Numerikus korrektség vizsgálata:**
* Összehasonlítás referencia implementációval: `numpy.polyfit(x, y, 1)` vagy `scipy.stats.linregress(x, y)`.
* Lebegőpontos számábrázolási eltérések elemzése (pl. Single vs. Double precision, numerikus stabilitás $N \sum x_i^2 - (\sum x_i)^2$ esetén).


* **6. Teljesítménymérés és kiértékelés:**
* Bemeneti méretek skálázása (pl. $10^4$-től $10^8$ pontig).
* Futási idők összevetése: CPU NumPy vs. Numba CPU vs. Numba GPU (külön bontva adatátviteli időre és tiszta kernel-futási időre).
* Gyorsulási tényező (Speedup) diagramok és magyarázatuk.

---

## Mért benchmark eredmények

**Mérési környezet:** NVIDIA Quadro P2000 (sm_61), driver 580.178.04, CUDA 12.4.131, Python 3.14.7, Numba 0.68.1, NumPy 2.5.4; mérés: 2026-11-08.

**Módszertan**

- Python: `fit_linear_regression_gpu` teljes hívása (H2D + kernel + D2H), 5 futás mediánja; CPU: vektorizált NumPy összegek, 3 futás mediánja; hiba a $\beta_1$, $\beta_0$ együtthatókban.
- C++: `least_squares.cu`, N=10 000 000, külön mért H2D / kernel / D2H idő; CPU referencia: skaláris `double` összegzés.
- A Python GPU idő pinelt staging pufferes Numba-másolást használhat, a C++ `cudaMemcpy` pageable `std::vector`-ból dolgozik, ezért a H2D idők nem összevethetők közvetlenül.

**Python / Numba (`least_squares.py`)**

| N | GPU teljes | CPU (NumPy összegek) | Gyorsulás | \|Δβ₁\| | \|Δβ₀\| |
|---:|---:|---:|---:|---:|---:|
| 100 000 | 0.92 ms | 0.20 ms | 0.2× | 1.07e-14 | 5.59e-13 |
| 1 000 000 | 2.72 ms | 3.08 ms | 1.1× | 2.13e-14 | 1.33e-12 |
| 10 000 000 | 22.86 ms | 41.13 ms | 1.8× | 5.33e-15 | 2.62e-13 |

**CUDA C++ (`least_squares.cu`) — N=10 000 000**

| H2D | Kernel | D2H | Teljes | \|Δβ₁\| | \|Δβ₀\| |
|---:|---:|---:|---:|---:|---:|
| 51.040 ms | 1.960 ms | 0.020 ms | 53.043 ms | 2.787e-12 | 1.670e-10 |

**Megjegyzések**

- A lineáris regresszió erősen adatmozgatás-korlátos: a kernel maga ~2 ms, a teljes időt a H2D dominálja.
- Kis N-nél a GPU lassabb a másolási overhead miatt; a keresztezés 1 millió pont körül van.
- A CPU referencia közvetlen összegzés, nem `np.polyfit`; a numerikus egyezést korábban `np.polyfit`-tal is ellenőriztük (~1e-13).
