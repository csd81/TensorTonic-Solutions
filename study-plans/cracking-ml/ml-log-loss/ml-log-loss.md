# <span style="font-size: 20px;">Log Loss (Binary Cross-Entropy)</span>

<span style="font-size: 14px;">Binary log loss measures the quality of probability predictions for outcomes that can be either 0 or 1. It rewards probabilities placed near the correct label and strongly penalizes confident predictions placed near the wrong label. The result is one non-negative number averaged across all samples.</span>

---

## <span style="font-size: 16px;">Labels and Predicted Probabilities</span>

<span style="font-size: 14px;">Each true label $y_i$ is either $0$ or $1$. Each prediction $p_i$ is the model's estimated probability that the label is $1$.</span>

* <span style="font-size: 14px;">A prediction near $1$ expresses confidence in the positive class.</span>
* <span style="font-size: 14px;">A prediction near $0$ expresses confidence in the negative class.</span>
* <span style="font-size: 14px;">A prediction of $0.5$ expresses equal uncertainty between the two classes.</span>

<span style="font-size: 14px;">These predictions are already probabilities. They are not unrestricted logits, so this problem does not apply sigmoid or softmax before computing the loss.</span>

---

## <span style="font-size: 16px;">Reading the Binary Log-Loss Formula</span>

<span style="font-size: 14px;">For $n$ samples, binary log loss is</span>

$$
L=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\ln(p_i)+(1-y_i)\ln(1-p_i)\right]
$$

<span style="font-size: 14px;">The two terms handle the two possible labels. Because $y_i$ can only be zero or one, exactly one term is active for each sample.</span>

<span style="font-size: 14px;">When the true label is $1$, the second term disappears:</span>

$$
L_i=-\ln(p_i)
$$

<span style="font-size: 14px;">The loss is small when $p_i$ is close to one and grows as $p_i$ approaches zero.</span>

<span style="font-size: 14px;">When the true label is $0$, the first term disappears:</span>

$$
L_i=-\ln(1-p_i)
$$

<span style="font-size: 14px;">The loss is small when $p_i$ is close to zero and grows as $p_i$ approaches one.</span>

<span style="font-size: 14px;">This single formula therefore evaluates the probability assigned to the correct outcome. A correct and confident probability receives very little loss. An uncertain probability receives a moderate loss. A wrong and confident probability receives a large loss.</span>

---

## <span style="font-size: 16px;">Why Confidence Changes the Penalty</span>

<span style="font-size: 14px;">The logarithm makes the penalty grow sharply near a confidently wrong boundary. For a positive label:</span>

* <span style="font-size: 14px;">Predicting $0.9$ gives $-\ln(0.9)\approx0.105$.</span>
* <span style="font-size: 14px;">Predicting $0.5$ gives $-\ln(0.5)\approx0.693$.</span>
* <span style="font-size: 14px;">Predicting $0.01$ gives $-\ln(0.01)\approx4.605$.</span>

<span style="font-size: 14px;">The prediction $0.01$ is not merely on the wrong side of a decision threshold. It claims near certainty that the positive event will not happen, so the loss is much larger.</span>

<span style="font-size: 14px;">A hard accuracy score would treat probabilities $0.51$ and $0.99$ as the same positive decision. Log loss distinguishes them because it evaluates the quality of the probability itself. This makes it useful when confidence matters, not only the final class choice.</span>

---

## <span style="font-size: 16px;">Why Probabilities Must Be Clipped</span>

<span style="font-size: 14px;">The natural logarithm is not finite at zero. A positive label with predicted probability $0$ requires $\ln(0)$, and a negative label with predicted probability $1$ requires $\ln(1-1)$. Either case would produce an undefined or infinite numerical result.</span>

<span style="font-size: 14px;">Before applying the formula, clamp every probability into</span>

$$
[\epsilon,1-\epsilon],\qquad \epsilon=10^{-15}
$$

<span style="font-size: 14px;">The clipped value is</span>

$$
\widetilde{p}_i=\min\left(1-\epsilon,\max(\epsilon,p_i)\right)
$$

<span style="font-size: 14px;">A supplied zero becomes $10^{-15}$, and a supplied one becomes $1-10^{-15}$. Values already inside the interval remain unchanged. The clipping amount is tiny enough that ordinary probabilities are unaffected, while boundary inputs remain finite.</span>

<span style="font-size: 14px;">Clipping must happen before either logarithm is evaluated. Replacing infinity after the calculation is too late because the invalid value may already have contaminated the mean.</span>

---

## <span style="font-size: 16px;">Worked Examples</span>

### <span style="font-size: 14px;">Mixed predictions</span>

<span style="font-size: 14px;">For labels $[1,0,1,1]$ and probabilities $[0.9,0.1,0.8,0.7]$, the probabilities assigned to the correct outcomes are $0.9$, $0.9$, $0.8$, and $0.7$. The per-sample losses are</span>

$$
-\ln(0.9),\quad -\ln(0.9),\quad -\ln(0.8),\quad -\ln(0.7)
$$

<span style="font-size: 14px;">Their mean is approximately $0.1976$.</span>

### <span style="font-size: 14px;">Completely uncertain predictions</span>

<span style="font-size: 14px;">When every prediction is $0.5$, both labels receive the same loss:</span>

$$
-\ln(0.5)\approx0.6931
$$

<span style="font-size: 14px;">The mean remains $0.6931$ regardless of how many labels are zero or one, because the model assigns equal probability to both outcomes.</span>

### <span style="font-size: 14px;">Boundary predictions</span>

<span style="font-size: 14px;">For labels $[1,0]$ and predictions $[1,0]$, clipping changes the probabilities only by $10^{-15}$. Both per-sample losses are extremely close to zero, and the required four-decimal result is $0.0$.</span>

---

## <span style="font-size: 16px;">Mean Reduction and Rounding</span>

<span style="font-size: 14px;">Compute every per-sample loss, take their arithmetic mean, convert the result to a Python float, and then round to four decimal places. Rounding individual samples first can change the final answer because small errors accumulate before averaging.</span>

<span style="font-size: 14px;">The negative sign is applied to the mean because logarithms of probabilities between zero and one are non-positive. Negating them produces a non-negative loss, with zero as the ideal limiting value.</span>

---

## <span style="font-size: 16px;">Implementation Order</span>

* <span style="font-size: 14px;">Convert labels and predicted probabilities to numeric arrays.</span>
* <span style="font-size: 14px;">Clip every probability to $[10^{-15},1-10^{-15}]$.</span>
* <span style="font-size: 14px;">Compute the positive-label and negative-label log terms element by element.</span>
* <span style="font-size: 14px;">Add the two terms, negate them, and take the mean across samples.</span>
* <span style="font-size: 14px;">Return the result as a Python float rounded to four decimal places.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Taking a logarithm before clipping.** Exact probabilities of zero or one can make one of the logarithms undefined.</span>
* <span style="font-size: 14px;">**Treating probabilities as logits.** The inputs already lie between zero and one, so applying another sigmoid changes their meaning and produces the wrong loss.</span>
* <span style="font-size: 14px;">**Using only the positive-label term.** Negative labels require $\ln(1-p)$ and must contribute to the average.</span>
* <span style="font-size: 14px;">**Forgetting the leading negative sign.** Logarithms of valid probabilities are non-positive, so omitting the sign produces a negative score instead of a loss.</span>
* <span style="font-size: 14px;">**Rounding too early.** Average the full-precision per-sample values before rounding the final result.</span>

---