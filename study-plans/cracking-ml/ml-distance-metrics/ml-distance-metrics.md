# <span style="font-size: 20px;">Distance Metrics</span>

<span style="font-size: 14px;">A distance function assigns one number to a pair of vectors. Smaller values mean the vectors are closer according to the chosen definition, while identical vectors usually have distance zero. Different metrics emphasize different aspects of the same pair, so the best choice depends on what closeness should mean for the data.</span>

---

## <span style="font-size: 16px;">Begin with Coordinate Differences</span>

<span style="font-size: 14px;">The two vectors have the same length, so each coordinate in $x$ is paired with the corresponding coordinate in $y$. Most metrics in this problem begin with the absolute coordinate differences</span>

$$
d_i=|x_i-y_i|
$$

<span style="font-size: 14px;">The metrics differ mainly in how they combine these differences:</span>

* <span style="font-size: 14px;">Euclidean squares the differences, sums them, and takes a square root.</span>
* <span style="font-size: 14px;">Manhattan adds the absolute differences directly.</span>
* <span style="font-size: 14px;">Chebyshev keeps only the largest absolute difference.</span>
* <span style="font-size: 14px;">Minkowski uses a parameter to control how strongly large differences dominate.</span>
* <span style="font-size: 14px;">Cosine distance compares vector direction rather than coordinate gaps alone.</span>

---

## <span style="font-size: 16px;">Euclidean Distance</span>

<span style="font-size: 14px;">Euclidean distance is the ordinary straight-line distance between two points. It is also called the $L_2$ distance:</span>

$$
d_{2}(x,y)=\sqrt{\sum_i(x_i-y_i)^2}
$$

<span style="font-size: 14px;">Squaring makes every contribution non-negative and gives larger coordinate gaps more influence. The square root returns the result to the original units of the features.</span>

<span style="font-size: 14px;">For $x=[1,2,3]$ and $y=[4,5,6]$, the differences are $[-3,-3,-3]$. Their squared values sum to $27$, so</span>

$$
d_2=\sqrt{27}\approx5.1962
$$

<span style="font-size: 14px;">Euclidean distance is sensitive to feature scale. A coordinate measured in thousands can dominate a coordinate measured between zero and one, even when both features are equally important. Real ML pipelines commonly standardize features before using this distance.</span>

---

## <span style="font-size: 16px;">Manhattan Distance</span>

<span style="font-size: 14px;">Manhattan distance adds the absolute coordinate differences. It is also called city-block or $L_1$ distance:</span>

$$
d_1(x,y)=\sum_i|x_i-y_i|
$$

<span style="font-size: 14px;">The name comes from moving along perpendicular streets rather than taking a straight diagonal path. Every coordinate contributes in direct proportion to its gap.</span>

<span style="font-size: 14px;">For $x=[1,2,3]$ and $y=[4,5,6]$, each absolute difference is $3$, giving</span>

$$
d_1=3+3+3=9
$$

<span style="font-size: 14px;">Compared with Euclidean distance, Manhattan distance does not square large gaps. It can therefore be less dominated by one unusually large coordinate, although it remains sensitive to feature scale.</span>

---

## <span style="font-size: 16px;">Chebyshev Distance</span>

<span style="font-size: 14px;">Chebyshev distance keeps only the largest absolute coordinate difference:</span>

$$
d_{\infty}(x,y)=\max_i|x_i-y_i|
$$

<span style="font-size: 14px;">This metric answers a specific question: what is the worst coordinate mismatch between the two vectors? All smaller differences are ignored once the maximum is known.</span>

<span style="font-size: 14px;">For $x=[1,5,3]$ and $y=[4,2,6]$, the absolute differences are $[3,3,3]$, so the Chebyshev distance is $3$. For differences $[1,7,2]$, it would be $7$.</span>

<span style="font-size: 14px;">Chebyshev distance is useful when closeness requires every coordinate to remain within a tolerance. One bad coordinate determines the entire result.</span>

---

## <span style="font-size: 16px;">Minkowski Distance</span>

<span style="font-size: 14px;">Minkowski distance is a family controlled by a parameter $p\geq1$:</span>

$$
d_p(x,y)=\left(\sum_i|x_i-y_i|^p\right)^{1/p}
$$

<span style="font-size: 14px;">The parameter changes how much large coordinate gaps dominate the result.</span>

* <span style="font-size: 14px;">When $p=1$, Minkowski is exactly Manhattan distance.</span>
* <span style="font-size: 14px;">When $p=2$, Minkowski is exactly Euclidean distance.</span>
* <span style="font-size: 14px;">As $p$ becomes large, the biggest coordinate difference matters increasingly more, approaching Chebyshev behavior.</span>

<span style="font-size: 14px;">For $x=[1,2,3]$, $y=[4,5,6]$, and $p=3$, all absolute differences equal $3$:</span>

$$
d_3=(3^3+3^3+3^3)^{1/3}=81^{1/3}\approx4.3267
$$

<span style="font-size: 14px;">The absolute value must be applied before raising a difference to $p$. Without it, negative differences can cancel positive ones or produce invalid results for non-integer values of $p$.</span>

---

## <span style="font-size: 16px;">Cosine Distance</span>

<span style="font-size: 14px;">Cosine distance compares direction. It begins with cosine similarity, which divides the dot product by the product of vector lengths:</span>

$$
\operatorname{cosine\_similarity}(x,y)=\frac{x\cdot y}{\|x\|_2\|y\|_2}
$$

<span style="font-size: 14px;">Distance is one minus similarity:</span>

$$
d_{\mathrm{cos}}(x,y)=1-\frac{x\cdot y}{\|x\|_2\|y\|_2}
$$

<span style="font-size: 14px;">Vectors pointing in the same direction have similarity $1$ and distance $0$, even when one vector is a scaled version of the other. Orthogonal vectors have similarity $0$ and distance $1$. Vectors pointing in opposite directions have similarity $-1$ and distance $2$.</span>

<span style="font-size: 14px;">For $x=[1,0,0]$ and $y=[0,1,0]$, the dot product is zero and both norms equal one. The vectors are orthogonal, so their cosine distance is $1$.</span>

<span style="font-size: 14px;">Cosine distance is useful when direction matters more than magnitude, such as comparing text or embedding vectors whose overall scale may vary.</span>

### <span style="font-size: 14px;">The zero-vector rule in this problem</span>

<span style="font-size: 14px;">A zero vector has norm zero, so the cosine formula would divide by zero and is mathematically undefined. This problem defines a specific fallback: return $0.0$ when either vector has zero norm. Follow that stored contract directly rather than attempting the division or substituting another library's convention.</span>

---

## <span style="font-size: 16px;">How the Metrics Compare</span>

<span style="font-size: 14px;">The same pair can receive different distances because every metric asks a different question.</span>

* <span style="font-size: 14px;">**Euclidean:** what is the straight-line separation?</span>
* <span style="font-size: 14px;">**Manhattan:** what is the total coordinate-by-coordinate separation?</span>
* <span style="font-size: 14px;">**Chebyshev:** what is the largest single coordinate separation?</span>
* <span style="font-size: 14px;">**Minkowski:** how should coordinate gaps be combined for a chosen $p$?</span>
* <span style="font-size: 14px;">**Cosine:** how different are the vector directions?</span>

<span style="font-size: 14px;">For $x=[1,2]$ and $y=[2,4]$, cosine distance is zero because the vectors point in the same direction. Euclidean and Manhattan distances are not zero because the coordinates and magnitudes differ. Neither result is wrong; the metrics encode different meanings of closeness.</span>

---

## <span style="font-size: 16px;">Implementation Order</span>

* <span style="font-size: 14px;">Pair corresponding coordinates from the equal-length input vectors.</span>
* <span style="font-size: 14px;">Dispatch on the requested metric name and apply only its formula.</span>
* <span style="font-size: 14px;">For cosine distance, compute both norms and return $0.0$ immediately if either norm is zero.</span>
* <span style="font-size: 14px;">Use $p$ only for Minkowski distance.</span>
* <span style="font-size: 14px;">Round the final numeric distance to four decimal places.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting absolute values.** Manhattan, Chebyshev, and Minkowski require non-negative coordinate gaps. Signed differences can cancel and produce an invalid distance.</span>
* <span style="font-size: 14px;">**Returning cosine similarity.** The requested result is $1-\text{similarity}$, not the normalized dot product itself.</span>
* <span style="font-size: 14px;">**Ignoring the zero-vector contract.** Check norms before division and return the problem's required $0.0$ fallback.</span>
* <span style="font-size: 14px;">**Taking the wrong Minkowski root.** Raise the sum to $1/p$, not to $p$.</span>
* <span style="font-size: 14px;">**Rounding intermediate differences.** Compute the complete metric at full precision and round only the final result.</span>

---