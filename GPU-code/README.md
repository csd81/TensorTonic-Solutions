 
Feladatleírás

A feladatot CUDA/C++, vagy Python programnyelvek használatával kell elkészíteni a tanult GPU
programozási technikák alkalmazásával. Az alább felsorolt algoritmusok közül kell egyet kiválasztani
és megvalósítani, majd numerikus pontosságát és futási idejét megvizsgálni. Saját probléma is
kidolgozható a gyakorlatvezetővel való egyéni egyeztetés után. Nyilvánosan elérhető könyvtárak az
algoritmusok bizonyos lépéseihez használhatók, viszont a teljes megoldást nem helyettesíthetik!
A dokumentáció irányterjedelme minimum 4 oldal, kötelező jelleggel tartalmaznia kell az alábbi
pontokat:

A probléma specifikációja. A vonatkozó irodalom áttekintése, milyen területeken, milyen célból
használják a kiválasztott algoritmust. Esetlegesen már létező párhuzamos implementációk és
párhuzamosítási módszerek áttekintése.

Az algoritmus szekvenciális változatának bemutatása. Mik az algoritmus fontosabb lépései
Lehetséges párhuzamosítási stratégiák és kihívások. Az irodalomkutatás és/vagy a szekvenciális
változat teljesítménymérése során mik az algoritmus teljesítménykritikus részei, azokat milyen
módon lehet párhuzamosítani.

Az algoritmus GPU változatának bemutatása. A GPU-n implementált lépések technikai leírása, az
alkalmazott módszerek indoklása.

Numerikus korrektség igazolása referencia implementáció segítségével. A referencia lehet saját
implementáció a matematikai leírás alapján, vagy nyilvánosan elérhető könyvtár.

Teljesítménymérés során kapott eredmények és kiértékelésük. A mérések alapján az algoritmus
teljesítményének kiértékelése különböző méretű bemenetekre, paraméterkombinációkra;
összehasonlítás a szekvenciális referenciával.

Választható feladatok

Algoritmusok digitális jelfeldolgozáshoz

Egy- és kétdimenziós konvolúció és korreláció digitális jelek szűréséhez (alul- és felüláteresztő FIR).
Short-time Fourier-transzformáció jelek idő-frekvenciatartománybeli elemzéséhez.
Digitális jelek teljesítményspektrumának becslése Welch-módszerrel.

Numerikus módszerek

Gauss-elimináció és LU dekompozíció lineáris egyenletrendszerek megoldásához.
Laplace- és Poisson-féle parciális differenciálegyenletek megoldása Jacobi és Gauss-Seidel (esetleg
Successive Overrelaxation) iterációkkal négyzetes kétdimenziós geometrián.
Szinguláris érték felbontás (SVD) tetszőleges párhuzamos módszerrel.

Algoritmusok adatfeldolgozáshoz

Lineáris regresszió legkisebb négyzetek módszerrel (egyenes illesztés egy változóra).
Párhuzamos k-means algoritmus klaszterezéshez folytonos változókkal, Euklideszi
távolságmetrikával.



