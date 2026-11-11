# <span style="font-size: 20px;">Check a Product-Chain Gradient</span>

<span style="font-size: 14px;">A gradient tells us how a final result changes when each input changes. In a computation with several operations, an input can affect the result through an intermediate value. The chain rule is what carries that influence through the graph.</span>

<span style="font-size: 14px;">The graph in this problem has two stages:</span>

$$
e=ab+c
$$

$$
L=ef
$$

<span style="font-size: 14px;">The inputs are $a$, $b$, $c$, and $f$. The value $e$ is an intermediate result, and $L$ is the final scalar loss. The task computes the gradient in two different ways, then compares them.</span>

---

## <span style="font-size: 16px;">The forward pass</span>

<span style="font-size: 14px;">Before thinking about gradients, compute the ordinary values from left to right. First form $e$, then multiply it by $f$ to obtain $L$. For a concrete example, let $a$ be $2$, $b$ be $-3$, $c$ be $10$, and $f$ be $-2$:</span>

$$
e=(2)(-3)+10=4
$$

$$
L=(4)(-2)=-8
$$

<span style="font-size: 14px;">Keeping the intermediate value is important because it appears directly in one of the gradients.</span>

---

## <span style="font-size: 16px;">Analytic gradients</span>

<span style="font-size: 14px;">Start at the final multiplication. Since $L=ef$, the local sensitivity of $L$ to $e$ is $f$, while the sensitivity to $f$ is $e$:</span>

$$
\frac{\partial L}{\partial e}=f
$$

$$
\frac{\partial L}{\partial f}=e
$$

<span style="font-size: 14px;">Now inspect $e=ab+c$:</span>

$$
\frac{\partial e}{\partial a}=b
$$

$$
\frac{\partial e}{\partial b}=a
$$

$$
\frac{\partial e}{\partial c}=1
$$

<span style="font-size: 14px;">The inputs $a$, $b$, and $c$ affect $L$ through $e$. The chain rule multiplies each local derivative inside $e$ by the downstream sensitivity $f$:</span>

$$
\frac{\partial L}{\partial a}=bf
$$

$$
\frac{\partial L}{\partial b}=af
$$

$$
\frac{\partial L}{\partial c}=f
$$

$$
\frac{\partial L}{\partial f}=e=ab+c
$$

<span style="font-size: 14px;">For the example, the analytic gradient in the required order is:</span>

$$
\left(\frac{\partial L}{\partial a}, \frac{\partial L}{\partial b}, \frac{\partial L}{\partial c}, \frac{\partial L}{\partial f}\right) = (6, -4, -2, 4)
$$

---

## <span style="font-size: 16px;">Numerical gradients</span>

<span style="font-size: 14px;">A numerical gradient asks the same questions by perturbing the inputs. Compute the baseline loss once, increase one input by $h$, recompute the entire loss, and divide the change in loss by $h$. For $a$:</span>

$$
\frac{\partial L}{\partial a}\approx
\frac{L(a+h,b,c,f)-L(a,b,c,f)}{h}
$$

<span style="font-size: 14px;">Repeat this independently for $b$, $c$, and $f$. Every estimate must begin from the original inputs. Perturbing an intermediate such as $e$ would check a different derivative and would skip the path from the selected input to that intermediate.</span>

<span style="font-size: 14px;">For this graph, each input appears linearly when the other inputs are fixed, so the forward differences should be extremely close to the analytic values. Floating-point subtraction can still produce tiny disagreement.</span>

---

## <span style="font-size: 16px;">What a gradient check tells you</span>

<span style="font-size: 14px;">The analytic and numerical lists should describe the same four sensitivities. A convenient summary is the largest absolute difference between corresponding entries:</span>

$$
\text{max error}
=
\max_i \left|g_i^{\text{analytic}}-g_i^{\text{numerical}}\right|
$$

<span style="font-size: 14px;">A very small maximum error is evidence that the analytic formulas and their ordering are correct. It is normal for the error to be small rather than exactly zero because the numerical calculation uses finite precision.</span>

<span style="font-size: 14px;">Gradient checking is primarily a verification technique. Numerical differentiation requires another forward evaluation for every input, while analytic backpropagation reuses the graph's local derivatives efficiently. Here, the numerical version acts as an independent reference for the formulas you derived.</span>

---

## <span style="font-size: 16px;">A reliable implementation order</span>

1. <span style="font-size: 14px;">Convert all inputs and $h$ to 64-bit floating-point values.</span>
2. <span style="font-size: 14px;">Compute the baseline intermediate and loss.</span>
3. <span style="font-size: 14px;">Build the analytic list in $a$, $b$, $c$, $f$ order.</span>
4. <span style="font-size: 14px;">Build the numerical list using four independent forward differences in the same order.</span>
5. <span style="font-size: 14px;">Compute the absolute difference for every pair and take the largest one.</span>
6. <span style="font-size: 14px;">Return the loss, both gradient lists, and the maximum disagreement.</span>

---

## <span style="font-size: 16px;">Common mistakes</span>

- <span style="font-size: 14px;">Using $b$, $a$, and $1$ as the final gradients forgets the downstream multiplier $f$.</span>
- <span style="font-size: 14px;">Setting the gradient for $f$ to $f$ is incorrect. The derivative of $ef$ with respect to $f$ is the other factor, $e$.</span>
- <span style="font-size: 14px;">Perturbing more than one input at once mixes multiple sensitivities.</span>
- <span style="font-size: 14px;">Changing the order of one gradient list makes the comparison meaningless.</span>
- <span style="font-size: 14px;">Expecting exact equality ignores ordinary floating-point rounding.</span>

<span style="font-size: 14px;">This small graph contains the central idea of backpropagation: compute values forward, then combine local derivatives with downstream influence to determine how each earlier input affects the final loss.</span>

---