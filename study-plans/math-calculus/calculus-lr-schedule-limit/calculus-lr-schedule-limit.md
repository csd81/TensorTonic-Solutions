<span style="font-size: 14px;">Limits are one of the most fundamental concepts in calculus, and they appear naturally throughout machine learning, particularly in the analysis of training dynamics. Understanding how quantities behave as they approach extreme values (very large iteration counts, very small step sizes) is essential for reasoning about whether an optimization algorithm will converge.</span>

## <span style="font-size: 14px;">What Is a Limit?</span>

<span style="font-size: 14px;">The limit of a function $f(t)$ as $t$ approaches some value $a$ (which can be $\infty$) is the value that $f(t)$ gets arbitrarily close to. Formally:</span>

$$
\lim_{t \to a} f(t) = L
$$

<span style="font-size: 14px;">means that for every $\epsilon > 0$, there exists a $\delta > 0$ (or threshold $N$ when $a = \infty$) such that $|f(t) - L| < \epsilon$ whenever $t$ is sufficiently close to $a$.</span>

<span style="font-size: 14px;">Key properties of limits that are used frequently:</span>

* <span style="font-size: 14px;">**Sum rule**: $\lim (f + g) = \lim f + \lim g$</span>
* <span style="font-size: 14px;">**Product rule**: $\lim (f \cdot g) = \lim f \cdot \lim g$</span>
* <span style="font-size: 14px;">**Quotient rule**: $\lim (f / g) = \lim f / \lim g$ provided $\lim g \neq 0$</span>
* <span style="font-size: 14px;">**Constant multiple**: $\lim (c \cdot f) = c \cdot \lim f$</span>

## <span style="font-size: 14px;">Learning Rate Schedules in Machine Learning</span>

<span style="font-size: 14px;">In gradient-based optimization, the **learning rate** $\alpha$ controls the step size when updating model parameters:</span>

$$
w_{t+1} = w_t - \alpha_t \nabla L(w_t)
$$

<span style="font-size: 14px;">where $w_t$ are the parameters at step $t$, $\nabla L(w_t)$ is the gradient of the loss, and $\alpha_t$ is the learning rate at step $t$.</span>

<span style="font-size: 14px;">A **learning rate schedule** defines how $\alpha_t$ changes over time. The choice of schedule has a profound effect on training behavior:</span>

* <span style="font-size: 14px;">**Too large**: the optimizer overshoots the minimum and may diverge</span>
* <span style="font-size: 14px;">**Too small**: convergence is painfully slow</span>
* <span style="font-size: 14px;">**Just right**: the optimizer converges efficiently to a good solution</span>

<span style="font-size: 14px;">Common learning rate schedules include:</span>

* <span style="font-size: 14px;">**Constant**: $\alpha_t = \alpha_0$ for all $t$</span>
* <span style="font-size: 14px;">**Step decay**: $\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / s \rfloor}$ where $\gamma < 1$ and $s$ is the step interval</span>
* <span style="font-size: 14px;">**Exponential decay**: $\alpha_t = \alpha_0 \cdot e^{-\lambda t}$</span>
* <span style="font-size: 14px;">**Inverse time decay**: $\alpha_t = \alpha_0 / (1 + k t)$ where $k > 0$ is the decay rate</span>
* <span style="font-size: 14px;">**Polynomial decay**: $\alpha_t = \alpha_0 / (1 + k t)^p$ for some power $p > 0$</span>

## <span style="font-size: 14px;">The Inverse Time Decay Schedule</span>

<span style="font-size: 14px;">The inverse time decay schedule is one of the simplest and most theoretically well-motivated schedules:</span>

$$
\alpha(t) = \frac{\alpha_0}{1 + k t}
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$\alpha_0 > 0$ is the initial learning rate</span>
* <span style="font-size: 14px;">$k > 0$ is the decay constant that controls how quickly the rate decreases</span>
* <span style="font-size: 14px;">$t \geq 0$ is the time step (iteration number)</span>

<span style="font-size: 14px;">At $t = 0$, we have $\alpha(0) = \alpha_0$. As $t$ increases, the denominator $1 + kt$ grows linearly, so $\alpha(t)$ decreases monotonically toward zero.</span>

### <span style="font-size: 14px;">Computing the Limit</span>

<span style="font-size: 14px;">To find $\lim_{t \to \infty} \alpha(t)$, we observe that as $t \to \infty$, the denominator $1 + kt \to \infty$. A positive constant divided by a quantity that grows without bound gives:</span>

$$
\lim_{t \to \infty} \frac{\alpha_0}{1 + k t} = 0
$$

<span style="font-size: 14px;">This result holds for any $\alpha_0 > 0$ and $k > 0$. The learning rate approaches zero, meaning the optimizer takes progressively smaller steps. This property is desirable because it allows the algorithm to "settle down" near a minimum rather than perpetually bouncing around it.</span>

## <span style="font-size: 14px;">Infinite Series and Partial Sums</span>

<span style="font-size: 14px;">An **infinite series** is the sum of infinitely many terms:</span>

$$
S = \sum_{t=0}^{\infty} a_t = a_0 + a_1 + a_2 + \cdots
$$

<span style="font-size: 14px;">We cannot add infinitely many numbers directly. Instead, we study the **partial sums**:</span>

$$
S_T = \sum_{t=0}^{T} a_t
$$

<span style="font-size: 14px;">and ask whether $\lim_{T \to \infty} S_T$ exists and is finite. If this limit is finite, the series **converges**. If $S_T \to \infty$, the series **diverges**.</span>

### <span style="font-size: 14px;">The Harmonic Series</span>

<span style="font-size: 14px;">The most famous divergent series is the **harmonic series**:</span>

$$
\sum_{t=1}^{\infty} \frac{1}{t} = 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \cdots = \infty
$$

<span style="font-size: 14px;">Despite the fact that $1/t \to 0$, the partial sums grow without bound. This can be shown by grouping terms:</span>

$$
\begin{aligned}
1 &+ \frac{1}{2} + \underbrace{\frac{1}{3} + \frac{1}{4}}_{\geq 1/2} \\
&+ \underbrace{\frac{1}{5} + \frac{1}{6} + \frac{1}{7} + \frac{1}{8}}_{\geq 1/2} + \cdots
\end{aligned}
$$

<span style="font-size: 14px;">Each group of $2^k$ terms sums to at least $1/2$, so the total sum exceeds any finite bound.</span>

<span style="font-size: 14px;">The partial sums of the harmonic series grow logarithmically. More precisely:</span>

$$
\sum_{t=1}^{T} \frac{1}{t} \approx \ln(T) + \gamma
$$

<span style="font-size: 14px;">where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant.</span>

### <span style="font-size: 14px;">The p-Series</span>

<span style="font-size: 14px;">A generalization of the harmonic series is the **p-series**:</span>

$$
\sum_{t=1}^{\infty} \frac{1}{t^p}
$$

<span style="font-size: 14px;">The behavior depends on $p$:</span>

* <span style="font-size: 14px;">If $p \leq 1$: the series **diverges** (like the harmonic series)</span>
* <span style="font-size: 14px;">If $p > 1$: the series **converges** to a finite value</span>

<span style="font-size: 14px;">For example:</span>

* <span style="font-size: 14px;">$\sum 1/t$ diverges ($p = 1$)</span>
* <span style="font-size: 14px;">$\sum 1/t^2 = \pi^2/6$ converges ($p = 2$)</span>
* <span style="font-size: 14px;">$\sum 1/t^3 \approx 1.202$ converges ($p = 3$)</span>

<span style="font-size: 14px;">This distinction between $p = 1$ (divergent) and $p = 2$ (convergent) is precisely what makes the inverse time decay schedule special for optimization.</span>

## <span style="font-size: 14px;">The Robbins-Monro Conditions</span>

<span style="font-size: 14px;">In 1951, Herbert Robbins and Sutton Monro published a foundational paper on **stochastic approximation**, which laid the theoretical groundwork for algorithms like stochastic gradient descent. They proved that under certain conditions on the step size sequence $\{\alpha_t\}$, a stochastic iterative algorithm converges to the true solution.</span>

<span style="font-size: 14px;">The **Robbins-Monro conditions** require the step size sequence to satisfy two properties simultaneously:</span>

### <span style="font-size: 14px;">Condition 1: The Sum of Step Sizes Diverges</span>

$$
\sum_{t=0}^{\infty} \alpha_t = \infty
$$

<span style="font-size: 14px;">This condition ensures that the algorithm has enough "energy" to reach the optimum from any starting point. If the sum were finite, the total distance the parameters could travel would be bounded, and the algorithm might get stuck far from the optimum.</span>

<span style="font-size: 14px;">Intuitively, if you take steps that shrink too quickly (e.g., exponential decay $\alpha_t = \alpha_0 \cdot e^{-\lambda t}$), the sum $\sum e^{-\lambda t} = 1/(1-e^{-\lambda})$ is finite, and the optimizer cannot explore the full parameter space.</span>

### <span style="font-size: 14px;">Condition 2: The Sum of Squared Step Sizes Converges</span>

$$
\sum_{t=0}^{\infty} \alpha_t^2 < \infty
$$

<span style="font-size: 14px;">This condition ensures that the noise introduced by stochastic gradients is eventually dampened. In SGD, the gradient estimate at each step has variance proportional to $\alpha_t^2$. If the sum of these variances is infinite, the accumulated noise overwhelms the signal, and the algorithm fails to converge.</span>

<span style="font-size: 14px;">A constant learning rate $\alpha_t = \alpha_0$ fails this condition because $\sum \alpha_0^2 = \infty$. This is why SGD with a constant learning rate oscillates around the minimum rather than converging to it.</span>

### <span style="font-size: 14px;">The Tension Between the Two Conditions</span>

<span style="font-size: 14px;">The two conditions create a delicate balance:</span>

* <span style="font-size: 14px;">The step sizes must shrink slowly enough that their sum diverges (condition 1)</span>
* <span style="font-size: 14px;">The step sizes must shrink fast enough that the sum of squares converges (condition 2)</span>

<span style="font-size: 14px;">This means the step sizes must decay at a rate that is "between" $O(1)$ (too slow, fails condition 2) and $O(1/t^p)$ with $p > 1$ (too fast, fails condition 1). The boundary case $\alpha_t \sim c/t$ is the slowest decay rate that satisfies both conditions.</span>

## <span style="font-size: 14px;">Verifying the Inverse Time Decay Schedule</span>

<span style="font-size: 14px;">Let us verify that $\alpha(t) = \alpha_0 / (1 + kt)$ satisfies both Robbins-Monro conditions.</span>

### <span style="font-size: 14px;">Condition 1 Verification</span>

<span style="font-size: 14px;">For large $t$, we have $\alpha(t) \approx \alpha_0 / (kt)$, so:</span>

$$
\begin{aligned}
\sum_{t=0}^{T} \frac{\alpha_0}{1 + kt} &\geq \sum_{t=1}^{T} \frac{\alpha_0}{1 + kt} \\
&\geq \sum_{t=1}^{T} \frac{\alpha_0}{2kt} = \frac{\alpha_0}{2k} \sum_{t=1}^{T} \frac{1}{t}
\end{aligned}
$$

<span style="font-size: 14px;">Since $\sum_{t=1}^{T} 1/t \to \infty$ (harmonic series), we conclude:</span>

$$
\sum_{t=0}^{\infty} \alpha(t) = \infty \quad \checkmark
$$

### <span style="font-size: 14px;">Condition 2 Verification</span>

<span style="font-size: 14px;">Squaring the learning rate:</span>

$$
\alpha(t)^2 = \frac{\alpha_0^2}{(1 + kt)^2}
$$

<span style="font-size: 14px;">For large $t$, $\alpha(t)^2 \approx \alpha_0^2/(kt)^2 = \alpha_0^2/(k^2 t^2)$. The sum:</span>

$$
\begin{aligned}
\sum_{t=0}^{\infty} \frac{\alpha_0^2}{(1 + kt)^2} &\leq \alpha_0^2 + \sum_{t=1}^{\infty} \frac{\alpha_0^2}{(kt)^2} \\
&= \alpha_0^2 + \frac{\alpha_0^2}{k^2} \cdot \frac{\pi^2}{6} < \infty
\end{aligned}
$$

<span style="font-size: 14px;">Since $\sum 1/t^2 = \pi^2/6$ is finite (a convergent p-series with $p = 2$), the sum of squared step sizes is also finite:</span>

$$
\sum_{t=0}^{\infty} \alpha(t)^2 < \infty \quad \checkmark
$$

<span style="font-size: 14px;">Both conditions are satisfied, confirming that the inverse time decay schedule is a theoretically valid choice for SGD convergence.</span>

## <span style="font-size: 14px;">Analysis of Other Schedules</span>

<span style="font-size: 14px;">It is instructive to see which other common schedules satisfy or violate the Robbins-Monro conditions.</span>

### <span style="font-size: 14px;">Constant Schedule: $\alpha_t = \alpha_0$</span>

* <span style="font-size: 14px;">$\sum \alpha_t = \infty$: condition 1 is satisfied</span>
* <span style="font-size: 14px;">$\sum \alpha_t^2 = \infty$: condition 2 is **violated**</span>
* <span style="font-size: 14px;">Result: SGD with constant learning rate does not converge to the exact optimum. It oscillates in a neighborhood whose size is proportional to $\alpha_0$.</span>

### <span style="font-size: 14px;">Exponential Decay: $\alpha_t = \alpha_0 \cdot \gamma^t$ with $0 < \gamma < 1$</span>

* <span style="font-size: 14px;">$\sum \alpha_t = \alpha_0 / (1 - \gamma) < \infty$: condition 1 is **violated**</span>
* <span style="font-size: 14px;">$\sum \alpha_t^2 = \alpha_0^2 / (1 - \gamma^2) < \infty$: condition 2 is satisfied</span>
* <span style="font-size: 14px;">Result: the optimizer may not reach the optimum because it "runs out of budget" too quickly. In practice, exponential decay still works well when the initial learning rate is large enough, but it lacks the theoretical guarantee of convergence.</span>

### <span style="font-size: 14px;">Polynomial Decay: $\alpha_t = \alpha_0 / (1 + kt)^p$</span>

<span style="font-size: 14px;">The behavior depends on the exponent $p$:</span>

* <span style="font-size: 14px;">If $p \leq 1/2$: $\sum \alpha_t = \infty$ and $\sum \alpha_t^2 = \infty$. Condition 2 fails.</span>
* <span style="font-size: 14px;">If $1/2 < p \leq 1$: $\sum \alpha_t = \infty$ and $\sum \alpha_t^2 < \infty$. Both conditions satisfied.</span>
* <span style="font-size: 14px;">If $p > 1$: $\sum \alpha_t < \infty$ and $\sum \alpha_t^2 < \infty$. Condition 1 fails.</span>

<span style="font-size: 14px;">The valid range is $1/2 < p \leq 1$. Our inverse time schedule uses $p = 1$, which is the boundary case and decays as slowly as possible while still satisfying both conditions.</span>

## <span style="font-size: 14px;">Numerical Computation of Partial Sums</span>

<span style="font-size: 14px;">While the theoretical analysis proves convergence properties in the limit, in practice we work with finite sums. Computing partial sums for a given number of steps $T$ allows us to:</span>

* <span style="font-size: 14px;">Observe the growth rate of $\sum \alpha_t$ (which should increase logarithmically)</span>
* <span style="font-size: 14px;">Observe the apparent convergence of $\sum \alpha_t^2$ (which should approach a finite limit)</span>
* <span style="font-size: 14px;">Compare different schedules empirically</span>

<span style="font-size: 14px;">For the inverse time schedule, the partial sum has a closed-form approximation using the digamma function $\psi$:</span>

$$
\begin{aligned}
\sum_{t=0}^{T} \frac{\alpha_0}{1 + kt} &= \frac{\alpha_0}{k} \left[ \psi\!\left(T + 1 + \frac{1}{k}\right) - \psi\!\left(\frac{1}{k}\right) \right]
\end{aligned}
$$

<span style="font-size: 14px;">However, for the purposes of numerical computation, directly evaluating the sum using an array of time steps is straightforward and efficient.</span>

<span style="font-size: 14px;">The key steps for numerical evaluation are:</span>

* <span style="font-size: 14px;">Create an array of time steps $t = 0, 1, 2, \ldots, T$</span>
* <span style="font-size: 14px;">Compute $\alpha(t) = \alpha_0 / (1 + kt)$ for each time step using vectorized operations</span>
* <span style="font-size: 14px;">Sum the resulting array to get $\sum \alpha_t$</span>
* <span style="font-size: 14px;">Square the array element-wise and sum to get $\sum \alpha_t^2$</span>

## <span style="font-size: 14px;">Why the Limit Matters for Training</span>

<span style="font-size: 14px;">The fact that $\lim_{t \to \infty} \alpha(t) = 0$ has direct practical consequences:</span>

* <span style="font-size: 14px;">**Early training**: large $\alpha(t)$ allows rapid exploration of the loss landscape and fast descent toward the basin of attraction</span>
* <span style="font-size: 14px;">**Late training**: small $\alpha(t)$ enables fine-grained adjustments near the minimum, reducing oscillation caused by stochastic gradient noise</span>
* <span style="font-size: 14px;">**Convergence guarantee**: the combination of divergent sum and convergent sum of squares ensures that the optimizer both reaches the optimum and settles at it</span>

<span style="font-size: 14px;">In modern deep learning, schedules such as cosine annealing and warmup followed by decay are more common than pure inverse time decay. However, the Robbins-Monro conditions remain the theoretical gold standard for understanding why learning rate decay is necessary for exact convergence of SGD.</span>

## <span style="font-size: 14px;">Connection to the Comparison Test</span>

<span style="font-size: 14px;">The verification of the Robbins-Monro conditions uses the **comparison test** for series. This is a fundamental tool in mathematical analysis:</span>

* <span style="font-size: 14px;">If $0 \leq a_t \leq b_t$ for all $t$ and $\sum b_t < \infty$, then $\sum a_t < \infty$ (convergence by comparison)</span>
* <span style="font-size: 14px;">If $0 \leq b_t \leq a_t$ for all $t$ and $\sum b_t = \infty$, then $\sum a_t = \infty$ (divergence by comparison)</span>

<span style="font-size: 14px;">We used this test implicitly when comparing $\alpha_0/(1+kt)$ with $\alpha_0/(2kt)$ for divergence (comparing from below) and $\alpha_0^2/(1+kt)^2$ with $\alpha_0^2/(kt)^2$ for convergence (comparing from above).</span>

### <span style="font-size: 14px;">The Limit Comparison Test</span>

<span style="font-size: 14px;">A closely related tool is the **limit comparison test**: if $a_t, b_t > 0$ and</span>

$$
\lim_{t \to \infty} \frac{a_t}{b_t} = L
$$

<span style="font-size: 14px;">where $0 < L < \infty$, then $\sum a_t$ and $\sum b_t$ either both converge or both diverge. For our schedule:</span>

$$
\lim_{t \to \infty} \frac{\alpha_0/(1+kt)}{1/t} = \lim_{t \to \infty} \frac{\alpha_0 t}{1 + kt} = \frac{\alpha_0}{k}
$$

<span style="font-size: 14px;">Since $0 < \alpha_0/k < \infty$ and $\sum 1/t$ diverges, we conclude $\sum \alpha_0/(1+kt)$ also diverges.</span>

## <span style="font-size: 14px;">Decay Rate and the Constant $k$</span>

<span style="font-size: 14px;">The decay constant $k$ controls how quickly the learning rate decreases:</span>

* <span style="font-size: 14px;">**Small $k$**: the learning rate decreases slowly. At step $t$, $\alpha(t) \approx \alpha_0$ for $t \ll 1/k$. This means the optimizer takes many large steps before the decay becomes noticeable.</span>
* <span style="font-size: 14px;">**Large $k$**: the learning rate decreases rapidly. By step $t = 1/k$, the learning rate has already halved. This leads to faster convergence but potentially slower initial progress.</span>

<span style="font-size: 14px;">The half-life of the learning rate (the step at which $\alpha(t) = \alpha_0/2$) is:</span>

$$
\frac{\alpha_0}{1 + k t_{1/2}} = \frac{\alpha_0}{2} \quad \Rightarrow \quad t_{1/2} = \frac{1}{k}
$$

<span style="font-size: 14px;">So the half-life is simply $1/k$. For example, if $k = 0.01$, the learning rate halves after 100 steps.</span>

## <span style="font-size: 14px;">Summary of Key Results</span>

<span style="font-size: 14px;">For the inverse time decay schedule $\alpha(t) = \alpha_0 / (1 + kt)$ with $\alpha_0 > 0$ and $k > 0$:</span>

* <span style="font-size: 14px;">$\lim_{t \to \infty} \alpha(t) = 0$: the learning rate vanishes</span>
* <span style="font-size: 14px;">$\sum_{t=0}^{\infty} \alpha(t) = \infty$: the total "budget" is infinite (divergent series)</span>
* <span style="font-size: 14px;">$\sum_{t=0}^{\infty} \alpha(t)^2 < \infty$: the accumulated noise is bounded (convergent series)</span>
* <span style="font-size: 14px;">Both Robbins-Monro conditions are satisfied, guaranteeing convergence of SGD</span>
* <span style="font-size: 14px;">The schedule decays like $O(1/t)$, which is the slowest possible rate that satisfies both conditions</span>
* <span style="font-size: 14px;">Partial sums of $\alpha(t)$ grow like $(\alpha_0/k) \ln(T)$</span>
