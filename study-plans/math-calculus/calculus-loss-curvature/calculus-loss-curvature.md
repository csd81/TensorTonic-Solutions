<span style="font-size: 14px;">In machine learning, a loss function measures how far a model's predictions are from the true targets. Training a model means minimizing this loss. First derivatives (gradients) tell the optimizer which direction to move. Second derivatives (curvature) tell the optimizer how far to move. Understanding both is essential for building and tuning any learning system.</span>

## <span style="font-size: 14px;">Why Derivatives of Loss Functions Matter</span>

<span style="font-size: 14px;">Every gradient-based optimizer relies on the first derivative of the loss with respect to the model's output. Gradient descent updates parameters by stepping in the direction that locally decreases the loss the fastest. But the first derivative alone does not capture the full geometry of the loss landscape. Two loss functions can have the same gradient at a point but behave very differently in the neighborhood of that point.</span>

<span style="font-size: 14px;">The second derivative captures this local behavior. It measures how quickly the gradient itself is changing - the curvature of the loss surface. High curvature means the gradient changes rapidly, so small steps are appropriate. Low curvature means the gradient is relatively stable, so larger steps are safe. This connection between curvature and step size is the foundation of second-order optimization methods like Newton's method, natural gradient descent, and the algorithms inside XGBoost.</span>

## <span style="font-size: 14px;">First Derivatives and Gradient Descent</span>

<span style="font-size: 14px;">Consider a scalar loss $L(\hat{y})$ that depends on a prediction $\hat{y}$. The first derivative $\frac{dL}{d\hat{y}}$ tells us how sensitive the loss is to changes in the prediction. In gradient descent, we update the prediction (or the parameters that produce it) by moving in the negative gradient direction:</span>

$$
\hat{y}_{\text{new}} = \hat{y} - \eta \frac{dL}{d\hat{y}}
$$

<span style="font-size: 14px;">where $\eta$ is the learning rate. If $\frac{dL}{d\hat{y}} > 0$, the loss increases as $\hat{y}$ increases, so we decrease $\hat{y}$. If $\frac{dL}{d\hat{y}} < 0$, we increase $\hat{y}$. The magnitude of the gradient tells us the steepness of the loss at the current point.</span>

<span style="font-size: 14px;">In a neural network, we do not directly update $\hat{y}$. Instead, we use the chain rule to propagate $\frac{dL}{d\hat{y}}$ backward through the network to compute $\frac{dL}{dw}$ for each weight $w$. The term $\frac{dL}{d\hat{y}}$ is therefore the starting point of backpropagation - the "seed" gradient that flows back through the computation graph.</span>

### <span style="font-size: 14px;">The Role of the First Derivative in Different Loss Functions</span>

<span style="font-size: 14px;">Different loss functions produce different gradient signals. Mean squared error produces gradients proportional to the residual, which means large errors receive large gradients. Cross-entropy produces gradients that grow without bound as predictions approach incorrect extremes, providing a strong corrective signal. Huber loss produces bounded gradients for large errors, making it robust to outliers. The choice of loss function shapes the training dynamics through its gradient.</span>

## <span style="font-size: 14px;">Second Derivatives and Curvature</span>

<span style="font-size: 14px;">The second derivative $\frac{d^2L}{d\hat{y}^2}$ measures the curvature of the loss function at a point. Geometrically, it describes how the tangent line rotates as we move along the curve. A large positive second derivative means the loss is highly curved (bowl-shaped), while a zero second derivative means the loss is locally linear (flat curvature).</span>

### <span style="font-size: 14px;">Connection to Newton's Method</span>

<span style="font-size: 14px;">Newton's method uses both the first and second derivatives to find optimal step sizes. Instead of taking a fixed-size step in the gradient direction, Newton's method solves for the step that would reach the minimum of a local quadratic approximation:</span>

$$
\begin{aligned}
L(\hat{y} + \Delta) &\approx L(\hat{y}) + \frac{dL}{d\hat{y}} \cdot \Delta + \frac{1}{2} \frac{d^2L}{d\hat{y}^2} \cdot \Delta^2
\end{aligned}
$$

<span style="font-size: 14px;">Setting the derivative of this approximation to zero:</span>

$$
\begin{aligned}
\frac{dL}{d\hat{y}} + \frac{d^2L}{d\hat{y}^2} \cdot \Delta &= 0 \\
&\implies \Delta = -\frac{dL / d\hat{y}}{d^2L / d\hat{y}^2}
\end{aligned}
$$

<span style="font-size: 14px;">The Newton step divides the gradient by the curvature. In regions of high curvature (where the loss bends sharply), the step is small. In regions of low curvature (where the loss is nearly flat), the step is large. This adaptive step size is what makes Newton's method converge much faster than gradient descent near a minimum.</span>

### <span style="font-size: 14px;">Curvature and Learning Rate Selection</span>

<span style="font-size: 14px;">The second derivative also provides guidance for choosing the learning rate in gradient descent. If the maximum curvature of the loss is $H_{\max}$, then gradient descent is stable only when the learning rate satisfies $\eta < 2/H_{\max}$. A loss function with high curvature requires a smaller learning rate, while a loss with low curvature allows larger steps. This explains why the same learning rate can work well for one loss function but cause divergence with another.</span>

## <span style="font-size: 14px;">Mean Squared Error: Derivatives</span>

<span style="font-size: 14px;">The mean squared error for a single prediction-target pair is:</span>

$$
L_{\text{MSE}}(\hat{y}) = (y - \hat{y})^2
$$

<span style="font-size: 14px;">Note that some formulations include a factor of $1/2$ for convenience, but the standard definition uses the squared error directly.</span>

### <span style="font-size: 14px;">First Derivative</span>

<span style="font-size: 14px;">Let $r = y - \hat{y}$ be the residual. Then $L = r^2$ and $\frac{dr}{d\hat{y}} = -1$. By the chain rule:</span>

$$
\frac{dL_{\text{MSE}}}{d\hat{y}} = 2r \cdot \frac{dr}{d\hat{y}} = 2(y - \hat{y})(-1) = -2(y - \hat{y}) = 2(\hat{y} - y)
$$

<span style="font-size: 14px;">The gradient is proportional to the residual and points away from the target. If $\hat{y} > y$, the gradient is positive, pushing the prediction downward. If $\hat{y} < y$, the gradient is negative, pushing the prediction upward. The magnitude grows linearly with the error.</span>

### <span style="font-size: 14px;">Second Derivative</span>

<span style="font-size: 14px;">Differentiating the first derivative with respect to $\hat{y}$:</span>

$$
\frac{d^2L_{\text{MSE}}}{d\hat{y}^2} = \frac{d}{d\hat{y}}[2(\hat{y} - y)] = 2
$$

<span style="font-size: 14px;">The second derivative of MSE is constant. This means MSE has uniform curvature everywhere - it is a perfect parabola. The loss surface bends equally whether the prediction is close to or far from the target. This constant curvature means a single learning rate works equally well regardless of the current error magnitude. It also means Newton's method for MSE reduces to a simple rescaling of the gradient by $1/2$.</span>

## <span style="font-size: 14px;">Binary Cross-Entropy: Derivatives</span>

<span style="font-size: 14px;">The binary cross-entropy loss for a single sample with target $y \in \{0, 1\}$ and predicted probability $\hat{y} \in (0, 1)$ is:</span>

$$
L_{\text{CE}}(\hat{y}) = -\left[y \ln(\hat{y}) + (1-y) \ln(1-\hat{y})\right]
$$

<span style="font-size: 14px;">This loss has a deep information-theoretic interpretation: it measures the number of extra bits needed to encode outcomes from the true distribution using the predicted distribution. Minimizing cross-entropy is equivalent to maximum likelihood estimation for the Bernoulli model.</span>

### <span style="font-size: 14px;">First Derivative</span>

<span style="font-size: 14px;">Differentiating term by term:</span>

$$
\begin{aligned}
\frac{dL_{\text{CE}}}{d\hat{y}} &= -\left[\frac{y}{\hat{y}} + (1-y) \cdot \frac{-1}{1-\hat{y}}\right] \\
&= -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}
\end{aligned}
$$

<span style="font-size: 14px;">This can be combined into a single fraction:</span>

$$
\begin{aligned}
\frac{dL_{\text{CE}}}{d\hat{y}} &= \frac{-y(1-\hat{y}) + (1-y)\hat{y}}{\hat{y}(1-\hat{y})} \\
&= \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}
\end{aligned}
$$

<span style="font-size: 14px;">The numerator $\hat{y} - y$ is the same residual as in MSE, but it is divided by $\hat{y}(1-\hat{y})$. When $\hat{y}$ is near 0 or 1, this denominator becomes very small, amplifying the gradient. This means cross-entropy provides a very strong corrective signal when the model makes a confident but wrong prediction.</span>

<span style="font-size: 14px;">Consider the case $y = 1$: then $\frac{dL}{d\hat{y}} = \frac{\hat{y} - 1}{\hat{y}(1-\hat{y})} = \frac{-1}{\hat{y}}$. As $\hat{y} \to 0$ (confidently wrong), the gradient $\to -\infty$, providing an extremely strong push. As $\hat{y} \to 1$ (correct), the gradient $\to -1$, a mild signal. This asymmetric behavior is precisely what we want for classification.</span>

### <span style="font-size: 14px;">Second Derivative</span>

<span style="font-size: 14px;">Starting from the separated form $\frac{dL}{d\hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$, we differentiate each term:</span>

$$
\frac{d^2L_{\text{CE}}}{d\hat{y}^2} = \frac{y}{\hat{y}^2} + \frac{1-y}{(1-\hat{y})^2}
$$

<span style="font-size: 14px;">For $y = 1$: $\frac{d^2L}{d\hat{y}^2} = \frac{1}{\hat{y}^2}$, which blows up as $\hat{y} \to 0$.</span>

<span style="font-size: 14px;">For $y = 0$: $\frac{d^2L}{d\hat{y}^2} = \frac{1}{(1-\hat{y})^2}$, which blows up as $\hat{y} \to 1$.</span>

<span style="font-size: 14px;">In both cases, the curvature becomes extreme when the model is confidently wrong. At $\hat{y} = 0.5$ with either target, $\frac{d^2L}{d\hat{y}^2} = 4$. Compared to MSE's constant curvature of 2, cross-entropy has higher curvature near the boundaries, but can have lower curvature in the interior (at $\hat{y} = 0.5$ with $y = 0.5$ it would be exactly 4, but typical binary targets give varying curvature).</span>

<span style="font-size: 14px;">The high curvature near wrong predictions means that second-order methods would take smaller steps in these regions - exactly the right behavior, since the loss landscape is changing rapidly there.</span>

## <span style="font-size: 14px;">Huber Loss: Derivatives</span>

<span style="font-size: 14px;">The Huber loss is a piecewise function that combines the best properties of MSE and absolute error:</span>

$$
L_{\text{Huber}}(\hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">The parameter $\delta$ controls the transition between quadratic and linear behavior. For small errors ($|y - \hat{y}| \le \delta$), the loss is quadratic like MSE. For large errors ($|y - \hat{y}| > \delta$), it grows linearly like absolute error. This makes Huber loss robust to outliers: large errors do not dominate the loss as they would with MSE.</span>

### <span style="font-size: 14px;">First Derivative</span>

<span style="font-size: 14px;">Let $r = y - \hat{y}$ with $\frac{dr}{d\hat{y}} = -1$.</span>

<span style="font-size: 14px;">In the quadratic region ($|r| \le \delta$):</span>

$$
L = \frac{1}{2}r^2 \implies \frac{dL}{dr} = r \implies \frac{dL}{d\hat{y}} = r \cdot (-1) = -(y - \hat{y}) = \hat{y} - y
$$

<span style="font-size: 14px;">In the linear region where $r > \delta$ (i.e., $y - \hat{y} > \delta$, meaning $\hat{y}$ is too low):</span>

$$
\begin{aligned}
L &= \delta r - \frac{1}{2}\delta^2 \\
&\implies \frac{dL}{dr} = \delta \\
&\implies \frac{dL}{d\hat{y}} = \delta \cdot (-1) = -\delta
\end{aligned}
$$

<span style="font-size: 14px;">In the linear region where $r < -\delta$ (i.e., $\hat{y} - y > \delta$, meaning $\hat{y}$ is too high):</span>

$$
\begin{aligned}
L &= -\delta r - \frac{1}{2}\delta^2 = \delta|r| - \frac{1}{2}\delta^2 \\
&\implies \frac{dL}{dr} = -\delta \implies \frac{dL}{d\hat{y}} = -\delta \cdot (-1) \\
&= \delta
\end{aligned}
$$

<span style="font-size: 14px;">Combining all cases:</span>

$$
\frac{dL_{\text{Huber}}}{d\hat{y}} = \begin{cases} \hat{y} - y & \text{if } |y - \hat{y}| \le \delta \\ -\delta & \text{if } y - \hat{y} > \delta \\ \delta & \text{if } \hat{y} - y > \delta \end{cases}
$$

<span style="font-size: 14px;">The gradient is clipped to $[-\delta, \delta]$. This gradient clipping is the key feature that makes Huber loss robust: no single outlier can produce a gradient larger than $\delta$, preventing catastrophic parameter updates.</span>

### <span style="font-size: 14px;">Second Derivative</span>

$$
\frac{d^2L_{\text{Huber}}}{d\hat{y}^2} = \begin{cases} 1 & \text{if } |y - \hat{y}| \le \delta \\ 0 & \text{if } |y - \hat{y}| > \delta \end{cases}
$$

<span style="font-size: 14px;">In the quadratic region, the curvature is 1 (constant, like a narrower MSE). In the linear regions, the curvature is 0 (the loss surface is flat). At the transition points $|y - \hat{y}| = \delta$, the second derivative is discontinuous - it jumps from 1 to 0. The function itself and its first derivative are continuous at these points (this is by design; the Huber loss was specifically constructed to be $C^1$ continuous), but the second derivative is not.</span>

<span style="font-size: 14px;">This zero curvature in the linear region means that second-order methods receive no useful curvature information for outlier points. Newton's method would attempt to divide by zero curvature, which is why practical implementations add a small regularization term to the denominator.</span>

## <span style="font-size: 14px;">Comparing Curvature Across Loss Functions</span>

<span style="font-size: 14px;">The three loss functions exhibit fundamentally different curvature profiles:</span>

<span style="font-size: 14px;">MSE has constant curvature of 2 everywhere. The loss surface is a perfect parabola, and the optimizer sees the same landscape geometry regardless of how good or bad the current prediction is.</span>

<span style="font-size: 14px;">Cross-entropy has variable curvature that depends on the predicted probability. Near $\hat{y} = 0.5$, the curvature is moderate. As $\hat{y}$ approaches 0 or 1, the curvature increases without bound. This means the loss surface is sharply curved near confident predictions, which makes the optimizer cautious near decision boundaries.</span>

<span style="font-size: 14px;">Huber loss has binary curvature: 1 inside the threshold and 0 outside. The optimizer has quadratic information for small errors but linear (zero-curvature) information for large errors.</span>

### <span style="font-size: 14px;">Implications for Optimization</span>

<span style="font-size: 14px;">These curvature differences explain why different loss functions require different optimizer configurations. With MSE, a single global learning rate is appropriate because the curvature never changes. With cross-entropy, adaptive optimizers like Adam perform better because the curvature varies by orders of magnitude across different predictions. With Huber loss, the optimizer must handle the transition between informative and uninformative regions gracefully.</span>

## <span style="font-size: 14px;">Connection to XGBoost</span>

<span style="font-size: 14px;">XGBoost (eXtreme Gradient Boosting) provides one of the most direct applications of both first and second derivatives of loss functions. In XGBoost, each new tree is fit to optimize a second-order Taylor expansion of the loss function. For each training sample $i$, the algorithm computes:</span>

$$
g_i = \frac{\partial L}{\partial \hat{y}_i}, \quad h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2}
$$

<span style="font-size: 14px;">These are the gradient ($g_i$) and Hessian ($h_i$) of the loss at the current prediction. The optimal leaf value for a set of samples $I$ in a leaf is:</span>

$$
w^* = -\frac{\sum_{i \in I} g_i}{\sum_{i \in I} h_i + \lambda}
$$

<span style="font-size: 14px;">where $\lambda$ is a regularization parameter. The gain from splitting a node is:</span>

$$
\begin{aligned}
\text{Gain} = \frac{1}{2}\bigg[&\frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} \\
&- \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda}\bigg] - \gamma
\end{aligned}
$$

<span style="font-size: 14px;">Both the leaf values and the split criterion depend on the ratio of the gradient sum to the Hessian sum. This means XGBoost is literally a second-order method that uses both derivatives you compute in this problem. When you implement custom loss functions in XGBoost, you must provide both $g_i$ and $h_i$ - the exact quantities this problem asks you to compute.</span>

<span style="font-size: 14px;">For MSE with $g_i = 2(\hat{y}_i - y_i)$ and $h_i = 2$, the leaf value simplifies to the mean of the residuals (scaled). For cross-entropy, the Hessian $h_i$ varies per sample, which means different samples contribute differently to the split gain - samples where the model is uncertain (high curvature) have more influence on tree structure than samples where the model is confident.</span>

## <span style="font-size: 14px;">Natural Gradient and Fisher Information</span>

<span style="font-size: 14px;">The second derivative of cross-entropy has a deep connection to the Fisher information matrix. For the Bernoulli model with parameter $p$, the Fisher information is:</span>

$$
I(p) = \mathbb{E}\left[\left(\frac{\partial \ln f(x;p)}{\partial p}\right)^2\right] = \frac{1}{p(1-p)}
$$

<span style="font-size: 14px;">This is exactly the second derivative of cross-entropy when $y$ takes the expected distribution. The natural gradient method preconditions the ordinary gradient by the inverse Fisher information, effectively taking a Newton step in the space of probability distributions rather than in parameter space. This connection between loss curvature and information geometry is one of the deepest results in optimization theory for machine learning.</span>

## <span style="font-size: 14px;">Practical Considerations</span>

### <span style="font-size: 14px;">Numerical Stability</span>

<span style="font-size: 14px;">Cross-entropy derivatives involve division by $\hat{y}$ and $1-\hat{y}$, which can cause numerical issues when predictions are very close to 0 or 1. In practice, predictions are clamped to a range like $[\epsilon, 1-\epsilon]$ where $\epsilon \approx 10^{-7}$. Without clamping, a single prediction of $\hat{y} = 10^{-15}$ could produce a second derivative of $10^{30}$, which would overflow or destabilize the optimizer.</span>

### <span style="font-size: 14px;">Choosing the Huber Delta</span>

<span style="font-size: 14px;">The $\delta$ parameter in Huber loss controls the trade-off between MSE-like behavior (for small errors) and robustness to outliers (for large errors). A large $\delta$ makes Huber loss behave more like MSE, while a small $\delta$ makes it behave more like absolute error. In practice, $\delta$ is often set to the median absolute deviation of the residuals, which provides a robust estimate of the "typical" error scale.</span>

### <span style="font-size: 14px;">When Curvature Is Zero</span>

<span style="font-size: 14px;">The Huber loss has zero second derivative in its linear region, which poses challenges for second-order methods. If all samples in a leaf have large errors, the Hessian sum approaches zero, and the optimal leaf value becomes ill-defined. This is why XGBoost implementations add a regularization parameter $\lambda > 0$ to the denominator - it prevents division by near-zero Hessians and acts as an implicit Bayesian prior on the leaf values.</span>

## <span style="font-size: 14px;">Summary of Derivative Formulas</span>

<span style="font-size: 14px;">For reference, the complete set of formulas used in this problem:</span>

<span style="font-size: 14px;">MSE: $L = (y-\hat{y})^2$</span>

$$
\frac{dL}{d\hat{y}} = 2(\hat{y} - y), \quad \frac{d^2L}{d\hat{y}^2} = 2
$$

<span style="font-size: 14px;">Cross-Entropy: $L = -[y\ln\hat{y} + (1-y)\ln(1-\hat{y})]$</span>

$$
\begin{aligned}
\frac{dL}{d\hat{y}} &= -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}, \\
\frac{d^2L}{d\hat{y}^2} &= \frac{y}{\hat{y}^2} + \frac{1-y}{(1-\hat{y})^2}
\end{aligned}
$$

<span style="font-size: 14px;">Huber: $L = \frac{1}{2}(y-\hat{y})^2$ if $|y-\hat{y}| \le \delta$, else $\delta|y-\hat{y}| - \frac{1}{2}\delta^2$</span>

$$
\frac{dL}{d\hat{y}} = \begin{cases} \hat{y} - y & |y-\hat{y}| \le \delta \\ \text{sign}(\hat{y}-y) \cdot \delta & \text{otherwise} \end{cases}
$$

$$
\frac{d^2L}{d\hat{y}^2} = \begin{cases} 1 & |y-\hat{y}| \le \delta \\ 0 & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">These formulas constitute the complete analytical toolkit needed for this problem. Computing them correctly for arrays of predictions and targets, and understanding what the curvature values tell us about optimization behavior, is the core challenge.</span>