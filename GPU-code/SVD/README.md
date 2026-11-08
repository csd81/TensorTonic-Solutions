**Szinguláris Érték Felbontás (SVD)** 

---

### 1. Elméleti háttér és a megfelelő algoritmus kiválasztása

Tetszőleges $A \in \mathbb{R}^{m \times n}$ mátrix felbontása:


$$A = U \Sigma V^T$$


ahol $U \in \mathbb{R}^{m \times m}$ és $V \in \mathbb{R}^{n \times n}$ ortogonálisak, $\Sigma \in \mathbb{R}^{m \times n}$ pedig a főátlójában a rendezett szinguláris értékeket ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$) tartalmazza.

#### Párhuzamosításra alkalmas algoritmusok összevetése GPU-n:

1. **One-Sided Jacobi SVD (Ajánlott Numba CUDA-hoz)**
* **Működés:** Az $A$ mátrix oszlopaira egymást követő ortogonális síkforgatásokat (Givens-forgatásokat) alkalmazunk úgy, hogy az oszlopok páronként ortogonálissá váljanak: $A' = A \cdot V$. Amikor $A'$ oszlopai ortogonálisak, $\sigma_j = \Vert{}a'_j\Vert{}_2$, és az ortonormált oszlopok adják $U$-t, az akkumulált forgatási mátrix pedig $V$-t.
* **Előny:** Természetes módon particionálható diszjunkt oszloppárokra (Round-Robin vagy tournament ütemezés), amelyek függetlenül, masszívan párhuzamosan forgathatók egyetlen GPU sweep alatt. Nincs szükség tridiagonalizálásra vagy bidiagonalizálásra.


2. **Two-Sided Jacobi SVD**
* Szimmetrikus sajátérték-problémára ($A^T A$) alkalmazható, de mindkét oldalról szorozni kell, ami nagyobb memóriasávszélesség-igényt és bonyolultabb szinkronizációt jelent.


3. **Bidiagonalizáció + QR / Divide & Conquer (pl. cuSOLVER)**
* Householder-tükrözésekkel bidiagonális alakra hozás, majd redukció. CPU-n rendkívül gyors, de egyedi Numba kernelként megírni a fázisváltások és a szekvenciális függőségek miatt lényegesen nehezebb és kevésbé skálázódik egyszerű kernelekkel.



---

### 2. A One-Sided Jacobi SVD architektúrája és párhuzamosítása

#### A. Matematikai lépések egy $(p, q)$ oszloppárra:

Két oszlop, $a_p$ és $a_q$ ortogonalizálása egy $2 \times 2$-es forgatási mátrixszal:


$$[a_p', a_q'] = [a_p, a_q] \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$


A forgatási szög kiszámítása a skaláris szorzatokból:


$$\alpha = a_p \cdot a_p = \Vert{}a_p\Vert{}^2, \quad \beta = a_q \cdot a_q = \Vert{}a_q\Vert{}^2, \quad \gamma = a_p \cdot a_q$$


A szög tangense ($\zeta = \frac{\beta - \alpha}{2\gamma}$, $t = \frac{\text{sgn}(\zeta)}{\vert{}\zeta\vert{} + \sqrt{1 + \zeta^2}}$), amiből adódik $c = \frac{1}{\sqrt{1+t^2}}$ és $s = t \cdot c$. Ha $\vert{}\gamma\vert{}$ kisebb egy megadott toleranciaküszöbnél ($\epsilon$), a forgatás átugorható.

#### B. Párhuzamosítási stratégia (Parallel Sweeps):

* Egy **sweep** során az összes lehetséges $(p, q)$ oszloppárt ($n(n-1)/2$ darab) ortogonalizálni kell.
* **Round-Robin párosítás:** $n$ oszlop esetén (feltéve, hogy $n$ páros; páratlan esetén virtuális zérus oszloppal kiegészítve) a párok feloszthatók $n-1$ al-lépésre (sub-sweep). Minden al-lépésben pontosan $n/2$ pár van, amelyek diszjunktak, így **egymástól teljesen függetlenül, párhuzamosan forgathatók**.
* **GPU Kernel struktúra:**
* **1. megközelítés (1 blokk = 1 oszloppár):** Egy thread block kap egy $(p, q)$ párt. A blokkon belüli szálak végzik a skalárszorzatok párhuzamos redukcióját (shared memory felhasználásával), kiszámítják $c$ és $s$ értékét, majd párhuzamosan frissítik az $a_p, a_q$ vektorok elemeit és a $V$ mátrix megfelelő oszlopait.
* **Memóriakezelés és Coalescing:** Mivel az oszlopok menti műveletek C-folytonos (row-major) mátrixoknál strided memóriahozzáférést eredményeznek, érdemes a mátrixot transzponált formában (oszlop-folytonosként tárolva, vagy expliciten $A^T$-ként kezelve) tartani a GPU memóriájában, így a szálak szomszédos memóriacímeket olvasnak (coalesced memory access).



---

### 3. Javasolt implementációs lépések (Numba CUDA keretrendszer)

1. **Szekvenciális CPU referencia:**
* Tiszta Python/NumPy alapú One-Sided Jacobi implementáció double precision (`float64`) pontossággal.
* Megállási feltétel: egy teljes sweep alatt végzett forgatások száma 0, vagy $\max \vert{}\gamma\vert{} < \epsilon$.


2. **GPU Numba kernelek:**
* `dot_product_and_norms_kernel`: oszloppárok skalárszorzatainak ($\alpha, \beta, \gamma$) kiszámítása shared memory redukcióval.
* `rotate_columns_kernel`: az oszlopvektorok és a $V$ mátrix frissítése a kiszámított $c, s$ paraméterekkel.
* *Host oldali vezérlés:* Python ciklus a sweep-ek és a Round-Robin indexpárok kezelésére, amíg a konvergencia be nem következik (vagy fix maximális sweep-szám eléréséig).


3. **Végső normalizálás:**
* A ciklus végén az oszlopok normái adják a szinguláris értékeket ($\Sigma$), az oszlopok normálása pedig a bal oldali szinguláris vektorokat ($U$).
* Csökkenő sorrendbe rendezés (NumPy `argsort` segítségével a végén host oldalon).



---

### 4. A dokumentáció javasolt felépítése (min. 4 oldal)

* **1. Probléma specifikációja és szakirodalom**
* SVD definíciója, alkalmazásai: dimenziócsökkentés (PCA), gépi tanulás (ajánlórendszerek, LSA), jelfeldolgozás, pszeudoinverz képzés.
* Korábbi GPU-s megoldások áttekintése: összehasonlítás a cuSOLVER (`gesvd`, `gesvdj`) bidiagonalizációs és Jacobi megközelítéseivel.


* **2. Szekvenciális algoritmus bemutatása**
* One-Sided Jacobi elméleti menete, Givens forgatások matematikája, konvergenciakritériumok.
* Folyamatábra vagy pszeudokód a szekvenciális sweep végrehajtásáról.


* **3. Párhuzamosítási stratégiák és kihívások**
* Párhuzamosítási gátak azonosítása: adatfüggőség az oszloppárok között; a Round-Robin tournament séma bemutatása a függőségek feloldására.
* Teljesítménykritikus pontok: skaláris szorzatok memóriasávszélesség-korlátja (memory-bound jelleg), shared memory redukció szerepe, nem összefüggő memóriacímzés (strided vs. coalesced access).


* **4. GPU változat bemutatása (Numba CUDA)**
* Kernel architektúra: grid/block dimenziók, warp divergencia minimalizálása a forgatási feltételeknél.
* Shared memory kihasználása az oszloppárok normáinak redukciójához.
* Device-to-Host adatmozgatások minimalizálása (az összes sweep és állapotfrissítés GPU memóriában marad, a konvergencia-flag optimalizált kezelése).


* **5. Numerikus korrektség igazolása**
* Összehasonlítás `numpy.linalg.svd` vagy `scipy.linalg.svd` referenciával.
* Metrikák:
* Rekonstrukciós hiba: $\Vert{}A - U \Sigma V^T\Vert{}_F / \Vert{}A\Vert{}_F$ (Frobenius-norma).
* Ortogonalitási hiba: $\Vert{}U^T U - I\Vert{}_\infty$ és $\Vert{}V^T V - I\Vert{}_\infty$.
* Szinguláris értékek relatív eltérése: $\max_i \vert{}\sigma_i - \sigma_i^{ref}\vert{} / \sigma_1^{ref}$.




* **6. Teljesítménymérés és kiértékelés**
* Mérési környezet hardveres és szoftveres specifikációja (GPU típus, CUDA verzió, Numba verzió, OS).
* Skálázódás vizsgálata: futásidő mátrixméret szerint ($m \times n$, pl. $128 \times 128$-tól $2048 \times 2048$-ig, vagy téglalap alakú mátrixok $m \gg n$).
* Gyorsulás (Speedup) grafikonok: Szekvenciális CPU vs. Numba GPU vs. cuBLAS/cuSOLVER (ha összehasonlítási alapként felveszed).
* GPU kihasználtság elemzése (pl. compute vs. memory throughput arányok a sweep-ek alatt).

---

## Mért benchmark eredmények

**Mérési környezet:** NVIDIA Quadro P2000 (sm_61), driver 580.178.04, CUDA 12.4.131, Python 3.14.7, Numba 0.68.1, NumPy 2.5.4; mérés: 2026-11-08.

**Módszertan**

- Python (`SVD.py`): one-sided Jacobi, CPU Jacobi vs. GPU teljes hívás (átvitelekkel), 128×64; a további méreteknél csak GPU idő. Hiba: rekonstrukció, ortogonalitás és szinguláris értékek vs. `np.linalg.svd`.
- C++ (`SVD.cu`): 128×64, 9 sweep; a `compute` a sweep kernelek ideje, a H2D a first-call CUDA kontextus/allokáció miatt nem steady-state.
- A Python GPU sok kisméretű kernelindítást végez (sweepenként 2×(n_work−1)); a C++ ugyanezt alacsonyabb indítási overheaddal futtatja.

**Python / Numba (`SVD.py`) — 128×64**

| CPU Jacobi | GPU teljes | Gyorsulás | Rekonstrukciós hiba | \|UᵀU−I\|∞ | \|VVᵀ−I\|∞ | σ relatív hiba |
|---:|---:|---:|---:|---:|---:|---:|
| 179.54 ms | 79.22 ms | 2.3× | 1.03e-14 | 6.14e-12 | 3.54e-14 | 7.95e-15 |

**Python GPU skálázódás**

| M×N | Sweepek | GPU idő | Rekonstrukciós hiba | σ relatív hiba |
|---:|---:|---:|---:|---:|
| 256×128 | 10 | 180.3 ms | 2.20e-14 | 1.37e-14 |
| 2048×64 | 9 | 130.1 ms | 9.55e-15 | 6.99e-15 |
| 512×512 | 13 | 1107.2 ms | 1.14e-13 | 7.36e-14 |

**CUDA C++ (`SVD.cu`) — 128×64**

| Sweepek | Compute | H2D* | D2H | Rekonstrukciós hiba | \|UᵀU−I\|∞ | \|VVᵀ−I\|∞ |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 5.979 ms | 36.702 ms | 0.032 ms | 9.71e-15 | 9.98e-13 | 1.58e-14 |

\* A H2D a CUDA kontextus és az allokációk első inicializálását is tartalmazza, ezért nem hasonlítható a steady-state adatátviteli időhöz.

**Megjegyzések**

- A C++ compute idő (5.98 ms) jóval alacsonyabb a Numba GPU teljes idejénél (79.22 ms), mert utóbbi a sok kernelindítás és az átvitelek költségét is tartalmazza.
- A 128×64-es CPU Jacobi referencia 179.54 ms; nagyobb mátrixokra a Python CPU referencia nem lett lemérve.
- Minden méretnél a rekonstrukciós hiba ~1e-13 alatt van, az ortogonalitási hibák ~1e-12 nagyságrendűek.
