A párhuzamos $k$-means klaszterezés GPU-s megvalósítása Pythonban, **Numba CUDA** környezetben egy klasszikus és kiválóan optimalizálható feladat, amely szinte tökéletesen illeszkedik a beadandó követelményeihez. Mivel a felhívás kifejezetten tiltja az MI által generált kész kódot és dokumentációt, az alábbi áttekintés az algoritmus felépítéséhez, a Numba CUDA architektúrához, a párhuzamosítási stratégiákhoz és a dokumentáció felépítéséhez nyújt mérnöki szintű támpontokat.

---

### 1. Az algoritmus szekvenciális lépései és szűk keresztmetszete

A Lloyd-féle standard $k$-means eljárás két fő fázis periodikus ismétléséből áll a konvergenciáig (vagy adott iterációszámig):

1. **Hozzárendelési fázis (Assignment step):**
* Minden egyes $x_i \in \mathbb{R}^d$ ($i = 1, \dots, N$) adatpontra kiszámítjuk az Euklideszi távolság négyzetét mind a $k$ darab centroidtól:

$$d^2(x_i, c_j) = \sum_{m=1}^d (x_{im} - c_{jm})^2$$


* Kiválasztjuk a minimális távolságú centroid indexét: $l_i = \arg\min_j d^2(x_i, c_j)$.
* *Számítási komplexitás:* $\mathcal{O}(N \cdot k \cdot d)$ iterációnként. Ha $N$ nagy (pl. $10^5 - 10^7$), ez a szakasz teszi ki a futásidő 95-99%-át. Mivel a pontok feldolgozása teljesen független egymástól, ez a lépés **kifejezetten masszívan párhuzamosítható (embarrassingly parallel)**.


2. **Frissítési fázis (Update step):**
* Az új centroidok kiszámítása a hozzárendelt pontok átlagaként:

$$c_j = \frac{1}{\vert{}S_j\vert{}} \sum_{x_i \in S_j} x_i$$


* Ez két részfeladat: a koordináták dimenziónkénti összegzése csoportonként, illetve a klaszterek létszámának ($\vert{}S_j\vert{}$) számlálása.
* *Jellege:* Csökkentési (reduction) művelet, ahol adatütközés (race condition) léphet fel.



---

### 2. GPU és Numba CUDA párhuzamosítási stratégiák

A megvalósítás során a következő technikai rétegeket és szinteket érdemes figyelembe venni:

#### A. Memóriaszervezés és adatátvitel (Host-Device I/O)

* **Egyszeri másolás:** A teljes adatmennyiséget ($X$) csak egyszer szabad felmásolni a GPU globális memóriájába a futás legelején (`cuda.to_device(X)`), és nem iterációnként.
* **Coalesced memóriahozzáférés:** Ha a pontokat $N \times d$ méretű C-folytonos mátrixban tárolod, a szomszédos szálak szomszédos memóriaelemeket fognak olvasni (ha egy szál egy teljes sort vagy pontot dolgoz fel, érdemes figyelni a szál-indexelés és a memóriacímek illeszkedésére a DRAM coalescing érdekében).

#### B. 1. Fázis: Hozzárendelés (Assignment Kernel)

* **Szálhozzárendelés:** 1 dimenziós grid/blokk struktúrával minden egyes GPU szál egy konkrét $x_i$ adatpontért felel:
```python
i = cuda.grid(1)
if i < N:
    # centroid távolságok számítása és minimum keresése

```


* **Centroidok tárolása (Fast Memory):** Mivel $k$ és $d$ tipikusan nem túl nagy ($k \le 128$, $d \le 64$), a centroidokat nem feltétlenül a lassabb globális memóriából érdemes újra és újra lekérni. Két hatékony alternatíva:
* **Shared memory:** A blokk szálai közösen behúzzák a centroidokat a megosztott memóriába az iteráció elején (`cuda.shared.array`).
* **2D thread mapping / Constant memory:** Kisebb méretnél a Numba `cuda.const.array_like` mechanizmusa is szóba jöhet a broadcast olvasások felgyorsítására.



#### C. 2. Fázis: Centroid-frissítés (Update Fázis)

A frissítés megvalósítására két járható út van:

1. **Host-oldali aggregáció (hibrid CPU-GPU megoldás):**
* Csak a klasztercímkéket ($l$) másolod vissza a CPU-ra, és a szummázást a CPU-n végzed el (pl. NumPy vagy SciPy segítségével).
* *Előny:* Rendkívül egyszerű és hibamentes.
* *Hátrány:* Minden iterációban PCI-Express másolási overhead lép fel, ami elnyomhatja a GPU sebességelőnyét kisebb iterációknál.


2. **Teljesen GPU-n futó aggregáció (Atomic műveletek):**
* A globális memóriában inicializálsz egy $k \times d$ méretű összegző és egy $k$ elemű számláló tömböt nullákkal.
* A szálak a hozzárendelés után atomi összeadással (`cuda.atomic.add`) növelik a megfelelő klaszter koordinátáit és a darabszámot.
* Egy második, kicsi kernel ($k$ szállal) kiszámolja az új pozíciókat: $c_{j, m} = \text{sum}_{j, m} / \text{count}_j$.
* *Figyelem:* Nagy párhuzamosság esetén az atomi műveletek torlódhatnak (contention), ha sok pont esik ugyanabba a klaszterbe, de átlagos méreteknél ez még mindig jóval gyorsabb, mint az iterációnkénti GPU $\leftrightarrow$ Host adatmozgatás.



---

### 3. A dokumentáció kötelező fejezeteinek felépítése

A megadott követelményrendszer alapján a minimum 4 oldalas jelentést az alábbi struktúrában érdemes felépíteni:

1. **A probléma specifikációja és szakirodalmi áttekintés**
* A $k$-means formális matematikai definíciója és célfüggvénye (Within-Cluster Sum of Squares, WCSS).
* Tipikus alkalmazási területek: képszegmentálás, vektorkvantálás, anomáliadetektálás, gépi látás előfeldolgozása.
* Párhuzamosítási előzmények a szakirodalomban (pl. CUDA-alapú megvalósítások, shared memory és warp-szintű redukciók összehasonlítása).


2. **Szekvenciális algoritmus és szűk keresztmetszetek**
* Lloyd-algoritmus pszeudokódja.
* Profilozás CPU-n: demonstráld (pl. `cProfile` vagy `time` mérésekkel), hogy az Euklideszi távolságok számítása viszi el a futási idő túlnyomó részét, így az a legfőbb gyorsítási célpont.


3. **Párhuzamosítási architektúra és kihívások**
* GPU szálhierarchia illesztése: szálak, blokkok méretezése (pl. 128, 256 vagy 512 szál/blokk).
* A memóriahozzáférési minták (globális vs. megosztott memória) és a szálak közötti divergencia minimalizálása.
* Az összegzési fázis versenyhelyzeteinek kezelése (atomi műveletek költsége vs. CPU-ra másolás).


4. **A GPU-s változat implementációja (Numba CUDA)**
* Kernel függvények szerkezete (grid-stride loopok vagy 1D lefedés).
* A memóriaterületek kezelése (`cuda.device_array`, buffer allokációk az iterációk előtt).


5. **Numerikus korrektség vizsgálata**
* Összevetés egy referencia implementációval: pl. a `scikit-learn` `KMeans` moduljával vagy a saját szekvenciális kódoddal.
* Mivel a $k$-means lokális optimumot talál és a kezdeti centroidok elhelyezkedése (pl. véletlen vs. $k$-means++) meghatározó, a korrektségi tesztet **azonos kezdeti centroidokkal** (rögzített seed) végezd el.
* Eltérések vizsgálata lebegőpontos pontosság mellett (FP32 vs. FP64 eltérések, maximális koordináta-hiba, WCSS értékek egyezése).


6. **Teljesítménymérések és kiértékelés**
* **Gyorsulási arány (Speedup):** $S = T_{\text{CPU}} / T_{\text{GPU}}$.
* Különböző adatméretek ($N = 10^4, 10^5, 10^6, 10^7$) és klaszterszámok ($k = 8, 16, 32, 64$).
* A kernel végrehajtási idő mérése külön a CUDA memóriamásolásoktól (`cuda.synchronize()` és `time.perf_counter()` precíz használatával a valós GPU idők rögzítéséhez).
* Grafikonok bemutatása: adatméret függvényében hogyan nő a GPU fölénye a CPU-val szemben (ahol kisebb méreteknél a GPU még alulmaradhat az overhead miatt, de nagy $N$-nél elérheti a többtíz- vagy százszoros gyorsulást).

---

## Mért benchmark eredmények

**Mérési környezet:** NVIDIA Quadro P2000 (sm_61), driver 580.178.04, CUDA 12.4.131, Python 3.14.7, Numba 0.68.1, NumPy 2.5.4; mérés: 2026-11-08.

**Módszertan**

- Python: `benchmark.py`, D=8, K=8, max_iters=15, 3 ismétlés mediánja. A „GPU kernel” az assign + zero + update kernelek ideje; a „GPU wall” a teljes host-oldali hívás; a CPU referencia float64-es NumPy Lloyd; a hiba a végső inertia relatív eltérése.
- C++: `kmeans.cu`, N=200 000, D=8, K=8, 15 iteráció. A `setup` a CUDA kontextus első inicializálását is tartalmazza, ezért a steady-state összehasonlításból kihagyandó.

**Python / Numba (`kmeans.py`, `benchmark.py`) — CPU vs. GPU**

| N | GPU kernel | GPU wall | CPU | Gyorsulás | Inertia rel. hiba |
|---:|---:|---:|---:|---:|---:|
| 10 000 | 4.10 ms | 10.44 ms | 78.14 ms | 7.49× | 1.78e-10 |
| 100 000 | 19.76 ms | 27.95 ms | 663.10 ms | 23.72× | 2.11e-06 |
| 1 000 000 | 172.31 ms | 200.08 ms | 6849.16 ms | 34.23× | 7.95e-09 |

**CUDA C++ (`kmeans.cu`) — N=200 000, D=8, K=8, 15 iteráció**

| Assign | Zero | Update | Kernel összesen | D2H | H2D | Teljes (setup nélkül) |
|---:|---:|---:|---:|---:|---:|---:|
| 4.709 ms | 0.072 ms | 21.501 ms | 26.282 ms | 0.258 ms | 0.517 ms | 27.7 ms |

**Megjegyzések**

- A GPU float32, a CPU referencia float64; ez magyarázza a kis inertia-eltéréseket.
- A C++ `setup` (~40.7 ms) a CUDA kontextus létrehozását tartalmazza, ezért nem része a fenti 27.7 ms-nak.
- 15 iteráció alatt egyik futás sem konvergált (`converged=false`); a mérés a fix iterációszámú futásra vonatkozik.
- A legnagyobb költség az update fázis (globális atomi műveletek); a shared-memory aggregáció a következő optimalizálási lépés.
