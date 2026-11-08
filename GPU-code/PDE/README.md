A 2D Laplace- és Poisson-egyenletek iteratív megoldása (véges differenciás módszerrel, 5-pontos stencil segítségével) kitűnő téma GPU párhuzamosításra Numba (`@cuda.jit`) környezetben. Mivel a feladatkiírás tiltja a közvetlen kódgenerálást, az alábbi strukturált áttekintés a matematikai hátteret, a párhuzamosítási architektúrát, a Numbás megvalósítási stratégiákat és a dokumentáció felépítését foglalja össze segítségként az önálló munkához.

---

### 1. Matematikai háttér és a diszkretizáció

A 2D Poisson-egyenlet általános alakja:


$$\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = f(x, y)$$


(Laplace-egyenlet esetén $f(x, y) = 0$).

Négyzetes rácson ($h = \Delta x = \Delta y$) a másodrendű centrális differenciákat alkalmazva az 5-pontos stencil:


$$\frac{u_{i+1, j} - 2u_{i,j} + u_{i-1,j}}{h^2} + \frac{u_{i, j+1} - 2u_{i,j} + u_{i, j-1}}{h^2} = f_{i,j}$$

Átrendezve az $u_{i,j}$ rácspontra:


$$u_{i,j} = \frac{1}{4} \left( u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - h^2 f_{i,j} \right)$$

---

### 2. Algoritmusok és párhuzamosítási stratégiák

#### A) Jacobi-iteráció

* **Működése:** A következő időlépés/iteráció ($k+1$) minden rácspontjának értéke kizárólag az előző iteráció ($k$) szomszédos értékeiből számítódik.
* **Párhuzamosíthatóság:** Triviálisan párhuzamosítható („embarrassingly parallel”). Nincs adathozzáférési adatfüggőség az adott iteráción belül.
* **Memóriakezelés (Ping-Pong pufferek):** Két külön tömb szükséges (`u_old`, `u_new`). Minden iteráció után meg kell cserélni a mutatókat (pointer swap), elkerülve a memóriamásolást.

#### B) Gauss-Seidel és Red-Black (vörös-fekete) színezés

* **A naiv Gauss-Seidel korlátja:** A standard szekvenciális Gauss-Seidel azonnal felülírja a rácspontokat és felhasználja a már frissített szomszédokat ($u_{i-1,j}^{(k+1)}$, $u_{i,j-1}^{(k+1)}$). Ez térbeli függőséget hoz létre, ami megakadályozza a közvetlen rácsszintű párhuzamosítást.
* **Red-Black (sakktábla) rendezés:**
* A rácsot sakktáblaszerűen felosztjuk: egy pont „piros”, ha $(i + j) \bmod 2 == 0$, és „fekete”, ha $(i + j) \bmod 2 == 1$.
* Mivel egy piros pont szomszédai kizárólag feketék (és fordítva), az 5-pontos stencil lépései függetlenek:
1. **Kernel 1:** Minden piros pont frissítése a fekete pontok értékeiből.
2. **Szinkronizáció** (GPU kernel return vagy grid barrier).
3. **Kernel 2:** Minden fekete pont frissítése a frissített piros pontok értékeiből.




* **SOR (Successive Overrelaxation):** A Red-Black sémára könnyen ráépíthető egy $\omega \in (1, 2)$ relaxációs paraméterrel:

$$u_{i,j}^{(k+1)} = (1 - \omega) u_{i,j}^{(k)} + \omega \cdot u_{i,j}^{\text{GS}}$$



Ez jelentősen csökkenti a konvergenciához szükséges iterációk számát.

---

### 3. Numba CUDA implementációs szempontok

A Numbában való megvalósítás során az alábbi architektúrális döntéseket érdemes figyelembe venni:

* **Tread- és Block-leképezés:**
* 2D dimenziójú blokkok (`blockdim = (16, 16)` vagy `(32, 8)`) és 2D grid használata (`cuda.grid(2)`).
* Határellenőrzés: a peremfeltételek (Dirichlet) rögzítettek, így a számítási kernel csak a belső pontokra fusson (`1 <= i < N-1` és `1 <= j < N-1`).


* **Memóriatranszferek minimalizálása:**
* A rácsot a konvergencia-ciklus előtt kell átvinni a GPU-ra (`cuda.to_device`), és csak a ciklus lefutása után szabad visszamásolni (`copy_to_host`). A GPU és Host közötti oda-vissza másolás minden iterációban tönkretenné a teljesítményt.


* **Shared Memory és Haló cellák:**
* A memóriasávszélesség a fő szűk keresztmetszet (memory-bound kernel). A blokkon belüli közös memóriába (`cuda.shared.array`) betöltött 2D csempe (tile) és annak 1 széles halórétege csökkenti a globális memóriaelérések számát.


* **Konvergenciavizsgálat (Residual reduction):**
* Ha a leállási feltétel reziduális norma ($\Vert{}u^{(k+1)} - u^{(k)}\Vert{}_\infty < \epsilon$), ezt nem érdemes minden egyes iterációban kiszámolni, mert a GPU globális redukció vagy host-másolás overheadje magas. Elegendő például minden 50. vagy 100. lépésben ellenőrizni a konvergenciát.



---

### 4. A dokumentáció javasolt vázlata (min. 4 oldal)

1. **Bevezetés és elméleti háttér (kb. 1 oldal):**
* A Poisson- és Laplace-egyenletek fizikai kontextusa (elektrosztatika, hővezetés, folyadékáramlás).
* A véges differenciák módszere (FDM), diszkretizáció 5-pontos stencillel.
* Szakirodalmi kitekintés: Stencil számítások GPU-n, memória-hozzáférési minták (coalescing), Red-Black színezés háttere.


2. **Szekvenciális algoritmus és referenciamodell (kb. 0.5 - 1 oldal):**
* A CPU-s Jacobi és Gauss-Seidel algoritmus pszeudokódja és lépései.
* Referencia-megoldás definiálása (pl. analitikus megoldás egy ismert forrástagú problémára, mint $f(x,y) = -2\pi^2 \sin(\pi x)\sin(\pi y)$, vagy NumPy/SciPy alapú CPU megvalósítás).


3. **GPU párhuzamosítási stratégia (kb. 1 - 1.5 oldal):**
* Párhuzamossági hierarchia illesztése: rácspontok leképezése szálakra és blokkokra.
* Ping-pong pufferek és adatcsere elkerülése Jacobi esetén.
* Függőségek feloldása Red-Black rendezéssel Gauss-Seidel és SOR esetén.
* Memóriahierarchia vizsgálata: globális memória vs. shared memory alkalmazása a stencil szomszédok elérésére.


4. **Numerikus korrektség és verifikáció (kb. 0.5 oldal):**
* CPU és GPU eredmények összehasonlítása azonos rácsméreten és iterációszámon ($L_\infty$ és $L_2$ hibák).
* Eltérések elemzése (lebegőpontos műveletek asszociativitásának hiánya, single vs. double precision hatása).


5. **Teljesítménymérés és kiértékelés (kb. 1 - 1.5 oldal):**
* Tesztkörnyezet specifikációja (CPU típus, GPU modell, compute capability, VRAM sávszélesség).
* Méretezhetőség: futási idők és gyorsulási tényezők (speedup) rácsméretek függvényében (pl. $256^2, 512^2, 1024^2, 2048^2, 4096^2$).
* Tiszta kernel futási idő vs. adatmozgatási idők (PCIe transfer overhead).
* Shared memory-s optimalizáció hatása a natív globális memóriás változathoz képest.  

### 1. Red-Black Gauss-Seidel és SOR indexelési logika

A klasszikus Gauss-Seidel iteráció során a rácspontok bejárása sorfolytonos: az $u_{i,j}^{(k+1)}$ kiszámításakor a bal oldali szomszéd ($u_{i-1,j}^{(k+1)}$) és a felső szomszéd ($u_{i,j-1}^{(k+1)}$) már az új időlépésből származik, míg a jobb és alsó szomszéd még a régiből ($u_{i+1,j}^{(k)}$, $u_{i,j+1}^{(k)}$). Ez a szigorú adatfüggőség ellehetetleníti a teljes rács egyidejű párhuzamos frissítését.

A **Red-Black (sakktábla) felbontás** ezt a függőséget szétbontja két független csoportra az 5-pontos stencil ortogonális tulajdonságát kihasználva:

$$\text{szín}(i, j) = (i + j) \bmod 2$$

* **Piros (Red) pontok:** $(i + j) \bmod 2 = 0$
* **Fekete (Black) pontok:** $(i + j) \bmod 2 = 1$

Mivel bármely rácspont négy közvetlen szomszédjának indexösszege ellentétes paritású:

* $(i \pm 1) + j = (i + j) \pm 1 \not\equiv (i + j) \pmod 2$
* $i + (j \pm 1) = (i + j) \pm 1 \not\equiv (i + j) \pmod 2$

Ez garantálja, hogy:

1. Egyetlen piros pont sem függ más piros pontoktól.
2. Egyetlen fekete pont sem függ más fekete pontoktól.

```
       j=0   j=1   j=2   j=3
i=0  [ R ] [ B ] [ R ] [ B ]
i=1  [ B ] [ R ] [ B ] [ R ]
i=2  [ R ] [ B ] [ R ] [ B ]
i=3  [ B ] [ R ] [ B ] [ R ]

```

---

#### Kétmenetes kernel végrehajtás és szinkronizáció

A teljes iterációs lépés két egymást követő fázisra bomlik. Mivel GPU-n a blokkok közötti globális szinkronizáció (grid synchronization) költséges vagy korlátozott, a legtisztább megoldás a két fázis különálló kernelhívásként való indítása a Host oldalról:

1. **Vörös fázis (Red sweep):**
* Minden aktív szál egy piros pontot dolgoz fel.
* Beolvassa a szomszédos **fekete** pontokat (melyek még a $k$-adik állapotban vannak).
* Kiszámítja az új piros értékeket: $u_{\text{red}}^{(k+1)}$.


2. **Globális szinkronizáció:**
* A vörös kernel lefutása és befejeződése implicit szinkronizációs pontként szolgál a GPU streamen.


3. **Fekete fázis (Black sweep):**
* Minden aktív szál egy fekete pontot dolgoz fel.
* Beolvassa a szomszédos **piros** pontokat (melyek már a frissített $(k+1)$-edik állapotban vannak).
* Kiszámítja az új fekete értékeket: $u_{\text{black}}^{(k+1)}$.



---

#### Successive Overrelaxation (SOR) kiterjesztés

A Successive Overrelaxation a Gauss-Seidel által számított köztes érték ($\tilde{u}_{i,j}$) és az előző állapot ($u_{i,j}^{(k)}$) súlyozott átlagát veszi egy $\omega \in (1, 2)$ relaxációs faktor segítségével:

$$\tilde{u}_{i,j} = \frac{1}{4} \left( u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - h^2 f_{i,j} \right)$$

$$u_{i,j}^{(k+1)} = (1 - \omega) u_{i,j}^{(k)} + \omega \cdot \tilde{u}_{i,j} = u_{i,j}^{(k)} + \omega \left( \tilde{u}_{i,j} - u_{i,j}^{(k)} \right)$$

* Ha $\omega = 1$: visszakapjuk a standard Gauss-Seidel iterációt.
* Optimális $\omega_{\text{opt}}$ paraméter $N \times N$-es rácson, Dirichlet-peremfeltételek mellett:

$$\omega_{\text{opt}} \approx \frac{2}{1 + \sin\left(\frac{\pi}{N}\right)}$$



Nagy rácsoknál ($N \ge 512$) $\omega_{\text{opt}} \approx 1.95 - 1.99$, ami nagyságrendekkel csökkenti a konvergenciához szükséges lépésszámot a sima Jacobi-hoz képest.

---

#### Szálleképezési stratégiák

A szálak és rácspontok hozzárendelésére két fő megközelítés létezik:

1. **Teljes rácslefedés elágazással (Full Grid Mapping):**
* A szálháló a teljes 2D tartományt lefedi (`cuda.grid(2)`).
* A kernel elején feltétel vizsgálja a paritást: `if (i + j) % 2 == color_phase: számolás`.
* **Előny:** Egyszerű indexelés, természetes 2D koordináta-kezelés.
* **Hátrány:** Divergencia a warp-on belül: a szálak pontosan 50%-a tétlen marad egy-egy fázisban, csökkentve az elméleti instrukció-kihasználtságot.


2. **Tömörített indexelés (Half-Grid Mapping):**
* Csak annyi szál indul, ahány piros (vagy fekete) pont van (azaz $N \times \frac{N}{2}$).
* A szál $t_x, t_y$ azonosítóiból a valódi $i, j$ koordináta direkt képlettel adódik:

$$j = 2 \cdot t_y + ((t_x + \text{color\_phase}) \bmod 2)$$


* **Előny:** 100%-os warp kihasználtság, nincs tétlen szál.
* **Hátrány:** Bonyolultabb coalesced memóriahozzáférés, mert a szomszédos szálak a memóriában nem közvetlenül egymás melletti címekre mutatnak.



---

### 2. Shared Memory csempézés (Tiling) és Halócellák kezelése

A stencil számítások erősen memória-sávszélesség korlátosak (**memory-bound**). Egy 5-pontos stencilnél minden belső pont kiszámításához 4 szomszédot kell beolvasni a globális memóriából. Csempézés nélkül a legtöbb pontot négyszer olvasná be a GPU a különböző szomszédos szálakon keresztül. A **Shared Memory (SMem)** csempézés célja, hogy a pontokat csak egyszer töltsük be a gyors chipen belüli memóriába.

---

#### Csempegeometria és Halóréteg

Tegyük fel, hogy a számítási blokk mérete $B_x \times B_y$ (például $16 \times 16 = 256$ szál).
A blokk által feldolgozott belső pontok tartománya: $B_x \times B_y$.
Az 5-pontos stencilhez azonban a határokon lévő szálaknak szükségük van a blokk szomszédságában lévő pontokra is. Ezért a Shared Memory-ban lefoglalt puffer mérete:

$$(B_x + 2) \times (B_y + 2)$$

* **Belső csempe (Interior Tile):** $B_x \times B_y$ méretű, a szálak tényleges számítási zónája.
* **Halócellák (Apron / Ghost cells):** 1 elem vastag keret a csempe körül (észak, dél, kelet, nyugat), amelyet szintén be kell tölteni a globális memóriából, de a jelenlegi blokk szálai nem frissítik az értéküket.

```
       +---+-------------------------+---+
       |   |       Északi Haló       |   |  <- y_smem = 0
       +---+-------------------------+---+
       | N |                         | K |
       | Y |                         | E |
       | U |     Számítási Zóna      | L |
       | G |     (Belső Csempe)      | E |
       | A |      Bx x By szál       | T |
       | T |                         | I |
       +---+-------------------------+---+
       |   |        Déli Haló        |   |  <- y_smem = By + 1
       +---+-------------------------+---+
         ^                             ^
      x_smem=0                      x_smem=Bx+1

```

---

#### A csempézett kernel végrehajtásának fázisai

A megvalósítás logikai lépései a csempézett blokkon belül:

1. **Lokális és globális koordináták meghatározása:**
* Globális indexek: $g_x = \text{blockIdx.x} \times B_x + \text{threadIdx.x}$, $g_y = \text{blockIdx.y} \times B_y + \text{threadIdx.y}$.
* Shared memory belső koordináták: $s_x = \text{threadIdx.x} + 1$, $s_y = \text{threadIdx.y} + 1$ (az eltolás helyet hagy a nyugati és északi halónak).


2. **Központi cellák betöltése:**
* Minden szál beemeli a saját koordinátájához tartozó globális elemet a megosztott memóriába:

$$\text{SMem}[s_x, s_y] \leftarrow \text{Global}[g_x, g_y]$$




3. **Halócellák együttműködő betöltése (Cooperative loading):**
A peremeken lévő szálak felelősek a szomszédos blokkok adatait képező halócellák beolvasásáért:
* **Nyugati haló ($s_x = 1$):** ha $\text{threadIdx.x} == 0$, beolvassa a bal oldali szomszédot:

$$\text{SMem}[0, s_y] \leftarrow \text{Global}[g_x - 1, g_y]$$


* **Keleti haló ($s_x = B_x$):** ha $\text{threadIdx.x} == B_x - 1$, beolvassa a jobb oldali szomszédot:

$$\text{SMem}[B_x + 1, s_y] \leftarrow \text{Global}[g_x + 1, g_y]$$


* **Északi haló ($s_y = 1$):** ha $\text{threadIdx.y} == 0$, beolvassa a felső szomszédot:

$$\text{SMem}[s_x, 0] \leftarrow \text{Global}[g_x, g_y - 1]$$


* **Déli haló ($s_y = B_y$):** ha $\text{threadIdx.y} == B_y - 1$, beolvassa az alsó szomszédot:

$$\text{SMem}[s_x, B_y + 1] \leftarrow \text{Global}[g_x, g_y + 1]$$



*(Megjegyzés: 5-pontos stencil esetén a 4 sarok halócella érintetlen maradhat, mert átlós szomszédokat nem használ a séma.)*


4. **Blokkszintű szinkronizáció:**
* Meg kell hívni a `cuda.syncthreads()` utasítást. Ez garantálja, hogy a blokk egyetlen szála se kezdje el a számolást addig, amíg az összes belső és halócellát be nem töltötte a warp-ok összessége.


5. **Számítás végrehajtása a Shared Memory-ból:**
* A szál kizárólag a gyors SMem-ből olvassa az 5 pontot:

$$\text{center} = \text{SMem}[s_x, s_y]$$


$$\text{north} = \text{SMem}[s_x, s_y - 1], \quad \text{south} = \text{SMem}[s_x, s_y + 1]$$


$$\text{west} = \text{SMem}[s_x - 1, s_y], \quad \text{east} = \text{SMem}[s_x + 1, s_y]$$


* Elvégzi a Poisson/SOR formulát (figyelembe véve a globális peremfeltételeket és a paritást).


6. **Eredmény visszaírása:**
* Az új értéket közvetlenül a globális memóriába írja vissza:

$$\text{Global\_Out}[g_x, g_y] \leftarrow u_{\text{new}}$$





---

### 3. Tervezési kompromisszum: SMem Csempézés vs. L1/L2 Cache

A modern GPU architektúrákon (NVIDIA Volta, Ampere, Ada Lovelace, Hopper/Blackwell) az L1 adat-gyorsítótár és a Shared Memory ugyanazon a fizikai SRAM blokkon osztozik. A globális memóriás direkt olvasásokat az L1 és L2 cache automatikusan gyorsítótárazza a szomszédos blokkok között.

| Szempont | Egyszerű Globális Memória + Cache | Kézi Shared Memory Csempézés |
| --- | --- | --- |
| **Kódkomplexitás** | Alacsony, minimális hibaforrás | Magas (haló-elágazások, indexeltolások, `syncthreads`) |
| **Szál-divergencia** | Nincs betöltési divergencia | A keretszálak extra elágazásokba futnak a halóknál |
| **Szinkronizációs költség** | Nincs blokkon belüli gát | Kötelező `cuda.syncthreads()` minden csempénél |
| **Sávszélesség hatékonyság** | Erősen támaszkodik az L2/L1 cache találati arányra | Determinisztikusan garantálja a minimális globális forgalmat |

A dolgozat mérési fejezetéhez kiváló kísérleti alap összehasonlítani a **natív globális memóriát használó Red-Black kernelt** a **Shared Memory-s változattal**: kisebb rácsméreteknél gyakran a sima globális kernel a gyorsabb az elágazások és szinkronizációk hiánya miatt, míg nagy méreteknél (ahol a munkahalmaz kiesik az L2 cache-ből) a csempézés előnye domborodik ki.


A Poisson-megoldók iterációs ciklusában a konvergencia ellenőrzése ($L_2$ vagy $L_\infty$ norma alapján) a teljesítmény egyik legkritikusabb szűk keresztmetszete. Ha minden egyes iteráció után kiszámítjuk a reziduumot, az két komoly problémát okoz:

1. **Memóriasávszélesség-pocsékolás:** Extra olvasási és írási körök a globális memóriában.
2. **Global Synchronization Bottleneck:** A teljes rácsra kiterjedő globális redukció (global reduction) miatt a GPU szálblokkoknak össze kell várniuk egymást, ami megöri a pipeline párhuzamosságot és magas kernel indítási (launch) overhead-et eredményez.

---

### 1. Fő optimalizációs stratégiák

#### A) Ritkított konvergenciavizsgálat (Periodic Checking)

A legegyszerűbb és legfontosabb mérnöki kompromisszum: **nem kell minden iterációban ellenőrizni a reziduumot.**

* Mivel a Jacobi/SOR iterációk konvergencia-görbéje egyenletes ütemben csökken, elegendő a reziduumot például **minden 50. vagy 100. lépésben** kiszámítani.
* Ezzel a globális redukció költsége 98-99%-ban eltűnik a futási időből.

#### B) $L_\infty$ norma előnyben részesítése $L_2$-vel szemben

A gyakorlatban parciális differenciálegyenleteknél szinte kizárólag az **$L_\infty$ normát** (a maximális abszolút hiba: $\max \vert{}r_{i,j}\vert{}$) használják leállási feltételként a következő okok miatt:

* **Matematikai művelet:** A `max` operátor asszociatív, kommutatív és idempotens, ráadásul nem érzékeny a lebegőpontos összeadások során fellépő kerekítési hibákra (kiesési hiba / cancellation), ellentétben az $L_2$ normával, ami négyzetösszeget és gyökvonást igényel.
* **GPU-barát implementáció:** Shared memory-ban vagy atomi műveletekkel (`atomicMax`) nagyságrendekkel olcsóbb egy globális maximumot keresni, mint egy pontos globális összegzést ($L_2$) végezni.

---

### 2. Kétlépcsős hierarchikus redukciós architektúra GPU-n

Ha mégis szükség van a reziduum kiszámítására, azt két fázisban érdemes megvalósítani, elkerülve a host-device adatmozgatást:

1. **1. Fázis: Blokk-szintű redukció (Kernel kimenet)**
* A reziduum-számító kernel minden szálban kiszámítja az adott rácspont hibáját: $r_{i,j} = \vert{}f_{i,j} - (\text{laplace}(u))\vert{}$.
* A szálak a blokkon belül a Shared Memory-t használva (vagy warp-szintű shuffle utasításokkal, pl. `cuda.shfl_down_sync`) felerősítik a blokk maximumát/összegét.
* Minden blokk kiírja a saját lokális eredményét egy kis méretű **részeredmény-tömbbe** (`partial_reductions`), amelynek mérete pontosan megegyezik a gridben lévő blokkok számával (pl. egy $4096 \times 4096$-os rácsnál $256 \times 256$ blokk esetén mindössze 65 536 elem).


2. **2. Fázis: Globális finalizáció (Global Finalize Kernel)**
* Egyetlen könnyű súlyú kernel fut le ezen a kis tömbön, amely véglegesíti a globális normát.
* Ezt az egyetlen skalár értéket masoljuk csak át a Host memóriába (vagy akár a GPU memóriában maradva a host oldalról egyetlen `cudaMemcpy`-vel olvasható ki, ha ritkított a vizsgálat).



---

### 3. Numba CUDA specifikus megvalósítási tippek

Numbában a warp-szintű műveletek (`cuda.shfl_down_sync`) és a shared memory tömbök (`cuda.shared.array`) kombinációjával rendkívül gyors $L_\infty$ redukció írható:

* **Atomikus max:** Ha a rácsméret nem túl gigantikus, a blokkok szintjén túllépve közvetlenül is dolgozhatunk egy globális változón atomikus művelettel (`cuda.atomic.max`), bár a shared memory redukció nagyságrenddel sávszélesség-hatékonyabb.
* **Memória-összevonás (Coalescing):** A reziduum-számító kernel olvasási mintája pontosan megegyezik az 5-pontos stencil olvasási mintájával, így a globális memóriából történő betöltés tökéletesen coalesced marad, ha a sorozatos indexelés érvényesül.

---

## Mért benchmark eredmények

**Mérési környezet:** NVIDIA Quadro P2000 (sm_61), driver 580.178.04, CUDA 12.4.131, Python 3.14.7, Numba 0.68.1, NumPy 2.5.4, SciPy 1.18.1; mérés: 2026-11-08.

**Fájlok:** `2D-Poisson-Jacobi.py` / `.cu` (naiv Jacobi) és `2D-Poisson-RedBlack-SOR.py` / `.cu` (optimalizált Red-Black SOR).

**Módszertan**

- Jacobi: fix 500 iteráció, `u_new` ping-pong pufferrel, float32; a GPU kernel idő `cuda.event`-tel mérve, egy reprezentatív futásból (bemelegítés után).
- SOR: float64, `ω = 2/(1+sin(π/N))`, in-place frissítés, a reziduum ellenőrzése 20 iterációnként, `tol = 1e-4`.
- H2D / D2H: `time.perf_counter()` + `cuda.synchronize()`.
- Numerikus ellenőrzés: Jacobi esetén N=256, 300 iteráció, vektorizált CPU Jacobi referenciával; SOR esetén N=256, a Python változat SciPy ritka direkt megoldással, a C++ változat szekvenciális CPU SOR referenciával.

### Python / Numba — naiv Jacobi (float32, 500 iteráció)

| N | H2D | Kernel | D2H | Total |
|---:|---:|---:|---:|---:|
| 256 | 0.65 ms | 20.47 ms | 0.10 ms | 21.22 ms |
| 512 | 0.95 ms | 25.70 ms | 0.56 ms | 27.21 ms |
| 1024 | 2.08 ms | 98.36 ms | 0.78 ms | 101.22 ms |

Ellenőrzés: `max |GPU − CPU| = 2.902e-09`.

### Python / Numba — Red-Black SOR (float64, tol=1e-4)

| N | ω | Iteráció | Reziduum | Kernel | Total | Konvergált |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.97575 | 720 | 6.43e-05 | 69.88 ms | 70.76 ms | igen |
| 512 | 1.98780 | 1500 | 8.04e-05 | 260.24 ms | 261.52 ms | igen |
| 1024 | 1.99388 | 3100 | 9.71e-05 | 1993.78 ms | 1997.97 ms | igen |

Ellenőrzés N=256: 800 iteráció, reziduum 8.345e-06; `max |GPU − SciPy| = 2.682e-09`.

### CUDA C++ — naiv Jacobi (float32, 500 iteráció)

| N | H2D | Kernel | D2H | Total |
|---:|---:|---:|---:|---:|
| 256 | 0.20 ms | 2.69 ms | 0.06 ms | 2.95 ms |
| 512 | 0.59 ms | 19.82 ms | 0.22 ms | 20.63 ms |
| 1024 | 3.15 ms | 81.61 ms | 0.60 ms | 85.36 ms |

Ellenőrzés N=256, 300 iteráció: `max |GPU − CPU| = 2.902e-09`.

### CUDA C++ — Red-Black SOR (float64, tol=1e-4)

| N | ω | Iteráció | Reziduum | Kernel | Total | Konvergált |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.97575 | 720 | 6.43e-05 | 25.48 ms | 25.90 ms | igen |
| 512 | 1.98780 | 1500 | 8.04e-05 | 242.58 ms | 243.57 ms | igen |
| 1024 | 1.99388 | 3100 | 9.71e-05 | 1968.03 ms | 1971.13 ms | igen |

Ellenőrzés N=256: GPU 800 iteráció, reziduum 8.345e-06; CPU referencia 1160 iteráció, reziduum 7.562e-10; `max |GPU − CPU| = 2.682e-09`.

**Kiértékelés**

- A naiv Jacobi nagyon lassan konvergál: N=256-on 5000 iteráció után a reziduum még mindig 0.957 (Python, 202.3 ms), miközben a Red-Black SOR 720 iteráció alatt 6.43e-05-re csökkenti (69.9 ms).
- A Red-Black SOR a valódi optimalizáció: az in-place frissítés és a gyorsabb konvergencia miatt nagyságrendekkel kevesebb iteráció kell, mint a Jacobihoz.
- A float32 SOR `ω ≈ 2` mellett megreked (N=256-on ~4.8e-2 reziduum), ezért a SOR implementációk float64-et használnak.
- A C++ Jacobi érezhetően gyorsabb a Numba változatnál (N=256: 2.69 ms vs. 20.47 ms), mert nincs JIT- és kernelindítási overhead. A C++ SOR N=256-on szintén gyorsabb (25.48 ms vs. 69.88 ms); N=1024-en a double precision számítási teljesítmény miatt a különbség eltűnik.
- A Jacobi és SOR eltérő konvergenciasebességű, ezért a fix iterációszámú Jacobi-idők nem összevethetők közvetlenül a SOR időkkel; a releváns összehasonlítás az azonos reziduumtolerancia eléréséhez szükséges idő.
