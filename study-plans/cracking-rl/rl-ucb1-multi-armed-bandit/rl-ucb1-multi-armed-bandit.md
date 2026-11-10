# <span style="font-size: 20px;">UCB1 for Multi-Armed Bandits</span>

<span style="font-size: 14px;">UCB1 (Auer, Cesa-Bianchi, and Fischer, 2002) is a deterministic bandit algorithm built on the principle of **optimism in the face of uncertainty**. Instead of flipping a coin to decide when to explore, it adds an **uncertainty bonus** to each arm's empirical mean and always picks the arm with the highest optimistic estimate. The bonus is large for arms that have been pulled rarely and shrinks as evidence accumulates, so exploration is steered precisely toward the arms the agent is least sure about. This directed exploration is what lets UCB1 achieve **logarithmic regret**, a dramatic improvement over the linear regret of fixed $\epsilon$-greedy.</span>

---

## <span style="font-size: 16px;">The Explore-Exploit Dilemma</span>

<span style="font-size: 14px;">A $K$-armed bandit presents $K$ actions, each with an unknown true mean reward $\mu_a$. The agent pulls one arm per round and observes only that arm's reward, so it must learn the means from its own samples. This is the core **exploration versus exploitation** tradeoff:</span>

* <span style="font-size: 14px;">**Exploitation** plays the arm with the best empirical mean to bank immediate reward, risking that an undersampled arm is actually better and was just unlucky early.</span>
* <span style="font-size: 14px;">**Exploration** plays an uncertain arm to tighten its estimate, accepting a likely lower immediate reward for information.</span>

<span style="font-size: 14px;">$\epsilon$-greedy resolves this with **undirected** randomness: it explores all non-greedy arms with equal probability, wasting pulls on arms already known to be bad. UCB1's insight is that exploration should be **directed** by how much an arm could plausibly be worth, given how little we have seen of it. An arm pulled twice deserves more benefit of the doubt than an arm pulled a thousand times.</span>

---

## <span style="font-size: 16px;">The UCB1 Score</span>

<span style="font-size: 14px;">For each arm $a$, UCB1 computes an upper confidence bound on its true mean and selects the arm that maximizes it:</span>

$$
\text{UCB}(a) = Q[a] + c \, \sqrt{\dfrac{\ln t}{N[a]}}
$$

<span style="font-size: 14px;">The two terms split cleanly into exploitation and exploration:</span>

* <span style="font-size: 14px;">**Exploitation term** $Q[a]$: the empirical mean reward of arm $a$, the running average of every reward observed from it. This rewards arms that have actually performed well.</span>
* <span style="font-size: 14px;">**Exploration bonus** $c\sqrt{\ln t / N[a]}$: an optimistic padding that represents how much the true mean could exceed the empirical mean. It grows slowly with the total step count $t$ through $\ln t$, and shrinks as the arm is pulled more through the $1 / \sqrt{N[a]}$ factor.</span>

<span style="font-size: 14px;">The constant $c > 0$ tunes how strongly exploration is favored. The classic UCB1 of Auer et al. fixes $c = \sqrt{2}$, giving the canonical bonus $\sqrt{2 \ln t / N[a]}$ that falls directly out of the Hoeffding concentration bound for rewards in $[0, 1]$. Larger $c$ explores more; smaller $c$ exploits more, recovering near-greedy behavior in the limit $c \to 0$.</span>

---

## <span style="font-size: 16px;">Why the Bonus Has This Shape</span>

<span style="font-size: 14px;">The bonus is not arbitrary; it is the half-width of a confidence interval derived from **Hoeffding's inequality**. For an arm whose rewards lie in $[0, 1]$ and have been averaged over $N[a]$ samples, Hoeffding bounds the probability that the empirical mean understates the true mean by more than $u$:</span>

$$
P\left(\mu_a > Q[a] + u\right) \le e^{-2 N[a] u^2}
$$

<span style="font-size: 14px;">Setting this failure probability to $t^{-4}$ and solving for $u$ gives $u = \sqrt{2 \ln t / N[a]}$, which is exactly the UCB1 bonus with $c = \sqrt 2$. The interpretation is that $\text{UCB}(a)$ is, with high probability, a true upper bound on $\mu_a$. Picking the arm with the largest upper bound means either that arm really is the best (a good exploit) or its bound is loose because it is underexplored (a good explore). Either way the algorithm makes progress, which is the elegance of optimism in the face of uncertainty.</span>

<span style="font-size: 14px;">The two factors in the bonus play distinct roles. The $\ln t$ in the numerator slowly inflates every arm's bonus as time passes, ensuring no arm is starved forever; even an arm that looks bad will eventually be revisited once enough rounds elapse. The $1 / \sqrt{N[a]}$ in the denominator drives the bonus toward zero for well-sampled arms, so the score converges to the pure empirical mean and the algorithm exploits.</span>

<span style="font-size: 14px;">A subtle but important detail is why $\ln t$ rather than $t$ or a constant. A constant bonus would be a fixed exploration tax that never adapts, much like fixed $\epsilon$-greedy, and would incur linear regret. A bonus growing like $\sqrt t$ would over-explore and also blow up the regret. The logarithm is the unique growth rate that explores just often enough to avoid being fooled by noise while still being slow enough that the cumulative exploration over $T$ rounds stays at the $O(\log T)$ order. This delicate balance is the technical heart of the Auer et al. proof: the failure probability $t^{-4}$ is chosen so that, summed over all rounds via $\sum_t t^{-4} < \infty$, the expected number of times the confidence bound is violated stays bounded by a small constant, contributing only the $\pi^2/3$ additive term to the regret.</span>

---

## <span style="font-size: 16px;">The Algorithm</span>

<span style="font-size: 14px;">UCB1 runs as a simple loop with one important initialization step:</span>

<span style="font-size: 14px;">1. **Initialize:** pull each arm once so that $N[a] \ge 1$ for all arms. This avoids division by zero in the bonus and gives every arm a finite starting estimate.</span>

<span style="font-size: 14px;">2. **Score:** at each round $t$, compute $\text{UCB}(a) = Q[a] + c\sqrt{\ln t / N[a]}$ for every arm.</span>

<span style="font-size: 14px;">3. **Select:** pull the arm $a_t = \arg\max_a \text{UCB}(a)$, breaking ties by a fixed rule such as lowest index.</span>

<span style="font-size: 14px;">4. **Update:** observe reward $R$, increment $N[a_t]$, and update the empirical mean incrementally:</span>

$$
Q[a_t] \leftarrow Q[a_t] + \frac{1}{N[a_t]}\left(R - Q[a_t]\right)
$$

<span style="font-size: 14px;">Unlike $\epsilon$-greedy and Thompson Sampling, UCB1 uses **no randomness at all** once the arms are initialized. Given the same history it always makes the same choice, which makes it easy to reason about and to reproduce.</span>

<span style="font-size: 14px;">The incremental mean update is worth dwelling on because it keeps the algorithm $O(1)$ in memory and time per arm. Rather than storing every reward and recomputing the average, it maintains only $Q[a]$ and $N[a]$ and nudges the estimate by the prediction error $R - Q[a]$ scaled by $1 / N[a]$. Early on the step size $1 / N[a]$ is large so estimates move quickly, and it tapers as evidence accumulates, mirroring the way the exploration bonus also shrinks with $N[a]$. The two mechanisms are aligned: an arm we have barely seen has both a volatile mean estimate and a large optimism bonus, and both stabilize together as the arm is sampled.</span>

---

## <span style="font-size: 16px;">Logarithmic Regret Bound</span>

<span style="font-size: 14px;">**Regret** is the cumulative reward lost relative to always playing the optimal arm with mean $\mu^*$:</span>

$$
\text{Regret}(T) = T\mu^* - \sum_{t=1}^{T}\mathbb{E}[\mu_{a_t}] = \sum_{a}\Delta_a\,\mathbb{E}[N_a(T)]
$$

<span style="font-size: 14px;">where $\Delta_a = \mu^* - \mu_a$ is the suboptimality gap of arm $a$. Auer et al. proved that UCB1 pulls each suboptimal arm only $O(\ln T / \Delta_a^2)$ times in expectation, which yields the bound:</span>

$$
\text{Regret}(T) \le \sum_{a : \Delta_a > 0}\frac{8 \ln T}{\Delta_a} + \left(1 + \frac{\pi^2}{3}\right)\sum_a \Delta_a
$$

<span style="font-size: 14px;">The leading term grows as $O(\log T)$, matching the Lai and Robbins (1985) asymptotic lower bound up to constants. This is the headline result: where fixed $\epsilon$-greedy incurs $\Theta(T)$ linear regret because it keeps sampling bad arms at a constant rate, UCB1 stops sampling a clearly inferior arm once its confidence interval no longer overlaps the best arm's. The intuition is that an arm with gap $\Delta_a$ is pulled only until its bonus $\sqrt{2 \ln t / N[a]}$ shrinks below roughly $\Delta_a$, which happens after $O(\ln t / \Delta_a^2)$ pulls. Smaller gaps are harder to detect and need more pulls, which is why $\Delta_a$ appears in the denominator.</span>

---

## <span style="font-size: 16px;">Choosing the Exploration Constant</span>

<span style="font-size: 14px;">The constant $c$ is the single dial that controls UCB1's appetite for exploration, and its effect is worth making concrete:</span>

* <span style="font-size: 14px;">**Theoretical value $c = \sqrt 2$.** This is the smallest constant for which the Hoeffding-based regret proof goes through for $[0, 1]$ rewards. It is the safe default when a worst-case guarantee matters.</span>
* <span style="font-size: 14px;">**Smaller $c$ (toward $0$).** The bonus shrinks, so the score is dominated by $Q[a]$ and the agent behaves almost greedily. This can win when the gaps $\Delta_a$ are large and easy to detect, but risks locking onto a lucky early leader.</span>
* <span style="font-size: 14px;">**Larger $c$.** The bonus dominates longer, so the agent spreads pulls more evenly. This helps when gaps are tiny and hard to resolve, at the cost of slower convergence to the best arm.</span>

<span style="font-size: 14px;">Because $\ln t$ grows without bound while $N[a]$ grows linearly for the chosen arm, the bonus for the eventual best arm decays toward zero and the empirical mean takes over. For the losing arms, $N[a]$ stops growing once they are abandoned, so their bonuses slowly re-inflate via $\ln t$ until they are briefly revisited; this self-correcting probing is what guarantees no arm is permanently starved.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Suppose $K = 3$ arms with empirical means $Q = [0.5,\ 0.6,\ 0.55]$, pull counts $N = [10,\ 2,\ 5]$, total steps $t = 17$, and $c = \sqrt 2 \approx 1.4142$. Note $\ln 17 \approx 2.833$.</span>

<span style="font-size: 14px;">1. **Arm 0:** bonus $= 1.4142\sqrt{2.833 / 10} = 1.4142\sqrt{0.2833} = 1.4142 \times 0.5323 \approx 0.7528$, so $\text{UCB}(0) \approx 0.5 + 0.7528 = 1.2528$.</span>

<span style="font-size: 14px;">2. **Arm 1:** bonus $= 1.4142\sqrt{2.833 / 2} = 1.4142\sqrt{1.4165} = 1.4142 \times 1.1902 \approx 1.6831$, so $\text{UCB}(1) \approx 0.6 + 1.6831 = 2.2831$.</span>

<span style="font-size: 14px;">3. **Arm 2:** bonus $= 1.4142\sqrt{2.833 / 5} = 1.4142\sqrt{0.5666} = 1.4142 \times 0.7527 \approx 1.0645$, so $\text{UCB}(2) \approx 0.55 + 1.0645 = 1.6145$.</span>

<span style="font-size: 14px;">Arm 1 wins despite the agent having pulled it only twice. Its modest empirical lead combined with its large uncertainty bonus pushes its optimistic estimate well above the others. This is optimism in action: the least-explored arm gets the most benefit of the doubt and is selected for another sample.</span>

<span style="font-size: 14px;">It is instructive to fast-forward this example. Suppose arm 1's true mean is in fact lower than arm 0's, and over the next several hundred rounds its empirical mean drifts down to $0.45$ while its count climbs to $N[1] = 300$. At $t = 1000$ its bonus becomes $1.4142\sqrt{\ln 1000 / 300} = 1.4142\sqrt{6.908/300} \approx 1.4142 \times 0.1517 \approx 0.2146$, so $\text{UCB}(1) \approx 0.45 + 0.2146 = 0.6646$. Meanwhile a genuinely better arm with mean $0.55$ and $N = 600$ would score about $0.55 + 1.4142\sqrt{6.908/600} \approx 0.55 + 0.1517 = 0.7017$ and overtake it. The interval has collapsed enough to reveal arm 1 as inferior, and the algorithm stops over-pulling it. This is exactly the mechanism that converts the early optimistic exploration into the eventual $O(\log T)$ regret.</span>

---

## <span style="font-size: 16px;">UCB1 in Context</span>

<span style="font-size: 14px;">UCB1 is the prototype of a whole family of confidence-bound methods and the bridge between simple bandits and modern RL exploration:</span>

* <span style="font-size: 14px;">**UCB-Tuned and KL-UCB** refine the bonus using observed reward variance or KL-divergence concentration, tightening constants and improving empirical performance over the plain Hoeffding bonus.</span>
* <span style="font-size: 14px;">**LinUCB** (Li et al., 2010) extends the optimism principle to contextual bandits, replacing the per-arm count with a confidence ellipsoid over a linear reward model.</span>
* <span style="font-size: 14px;">**UCT and the PUCT variant in AlphaGo and MuZero** apply UCB-style optimism to the action edges of a Monte Carlo tree search, demonstrating that the same confidence-bound idea scales to enormous decision spaces.</span>

<span style="font-size: 14px;">Compared to its peers in this section, UCB1 trades the value-aware-but-uncertainty-blind exploration of softmax and the Bayesian sampling of Thompson Sampling for a fully deterministic, frequentist guarantee. It is the easiest method to analyze and one of the easiest to defend in an interview because its regret bound is explicit and its behavior is reproducible.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Division by zero when an arm has never been pulled.** The bonus $\sqrt{\ln t / N[a]}$ is undefined for $N[a] = 0$. The standard fix is to pull each arm once during initialization, or to treat an unpulled arm as having infinite UCB so it is selected immediately.</span>
* <span style="font-size: 14px;">**Using the wrong time variable in $\ln t$.** The numerator must use the total number of rounds played so far, not the count for the specific arm. Plugging in $N[a]$ where $t$ belongs collapses the exploration term and breaks the regret guarantee.</span>
* <span style="font-size: 14px;">**Forgetting the reward-scale assumption.** The $c = \sqrt 2$ constant is derived assuming rewards lie in $[0, 1]$. If rewards have a different range, the bonus must be rescaled or $c$ retuned, otherwise the confidence intervals are mis-sized and exploration is either too timid or too aggressive.</span>
* <span style="font-size: 14px;">**Treating UCB1 as stochastic.** UCB1 is deterministic given the history. Adding extra randomness or expecting run-to-run variation misunderstands the method; its appeal is precisely that the optimism bonus, not a coin flip, drives all exploration.</span>

---