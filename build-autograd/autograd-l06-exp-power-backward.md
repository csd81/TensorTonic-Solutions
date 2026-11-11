# <span style="font-size: 20px;">Differentiate Exponentials and Powers</span>

<span style="font-size: 14px;">Exponential and power operations appear throughout optimization and neural networks. This problem evaluates both operations at one scalar input and combines each local derivative with the same upstream gradient.</span>

---

## <span style="font-size: 16px;">The exponential operation</span>

<span style="font-size: 14px;">The natural exponential maps an input $x$ to:</span>

$$
e=\exp(x)
$$

<span style="font-size: 14px;">Its derivative has the unusual property of being equal to the forward output:</span>

$$
\frac{de}{dx}=\exp(x)=e
$$

<span style="font-size: 14px;">If $u$ is the gradient arriving from later computation, the complete backward contribution is:</span>

$$
g_{\exp}=u\exp(x)=ue
$$

<span style="font-size: 14px;">Saving the forward exponential output therefore provides exactly the value needed during the backward calculation.</span>

---

## <span style="font-size: 16px;">The constant-power operation</span>

<span style="font-size: 14px;">A constant exponent $n$ defines:</span>

$$
p=x^n
$$

<span style="font-size: 14px;">The power rule multiplies by the exponent and reduces the exponent by one:</span>

$$
\frac{dp}{dx}=nx^{n-1}
$$

<span style="font-size: 14px;">After applying the same upstream gradient:</span>

$$
g_{\mathrm{power}}=unx^{n-1}
$$

<span style="font-size: 14px;">Unlike the exponential derivative, the power derivative usually cannot be recovered from the output alone without also knowing the input or exponent.</span>

---

## <span style="font-size: 16px;">Why the upstream gradient matters</span>

<span style="font-size: 14px;">The local derivatives describe only the immediate operation. The upstream value $u$ describes how strongly the final loss depends on that operation's output. The chain rule requires both factors.</span>

<span style="font-size: 14px;">A zero upstream gradient makes both returned gradients zero, regardless of the local slopes. A negative upstream gradient reverses their signs. The forward outputs remain unchanged because upstream information belongs only to the backward pass.</span>

---

## <span style="font-size: 16px;">The zero-exponent case</span>

<span style="font-size: 14px;">A zero exponent defines a constant function:</span>

$$
x^0=1
$$

<span style="font-size: 14px;">Since the output does not change with $x$, its derivative is zero:</span>

$$
\frac{d}{dx}x^0=0
$$

<span style="font-size: 14px;">This task defines the output as one and the gradient as zero for every finite base, including zero. Handling this case directly avoids trying to evaluate $x^{-1}$ inside the general derivative expression.</span>

---

## <span style="font-size: 16px;">Negative exponents and negative bases</span>

<span style="font-size: 14px;">A negative exponent represents a reciprocal power. For example:</span>

$$
x^{-2}=\frac{1}{x^2}
$$

$$
\frac{d}{dx}x^{-2}=-2x^{-3}
$$

<span style="font-size: 14px;">At $x=2$, the power output is $0.25$. With upstream gradient $-0.5$, the local derivative is $-0.25$, so the upstream-scaled gradient is $0.125$.</span>

<span style="font-size: 14px;">A negative base is valid here only when the exponent is integer-valued. Fractional powers of a negative real number may not have a real-valued result, which would violate the scalar real-number contract.</span>

---

## <span style="font-size: 16px;">A negative-base example</span>

<span style="font-size: 14px;">Take $x=-2$, exponent $n=3$, and upstream gradient $u=2$. The forward power is:</span>

$$
p=(-2)^3=-8
$$

<span style="font-size: 14px;">The power gradient is:</span>

$$
g_{\mathrm{power}}
=
2\cdot 3\cdot(-2)^2
=
24
$$

<span style="font-size: 14px;">The exponential uses the same input but remains positive:</span>

$$
\exp(-2)\approx 0.135335
$$

<span style="font-size: 14px;">Its upstream-scaled gradient is approximately $0.270671$.</span>

---

## <span style="font-size: 16px;">Numerical range</span>

<span style="font-size: 14px;">The exponential grows rapidly for large positive inputs and can overflow 64-bit floating point. For large negative inputs, it approaches zero and may underflow. Power operations can also overflow when a large magnitude is raised to a high exponent.</span>

<span style="font-size: 14px;">Valid inputs keep all four results finite in 64-bit floating point. Overflowing or undefined powers are outside this problem’s input domain.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting upstream scaling.** The returned gradients are chain-rule contributions, not only local derivatives.</span>
* <span style="font-size: 14px;">**Using the power output as its derivative.** That identity belongs to the exponential, while powers follow $nx^{n-1}$.</span>
* <span style="font-size: 14px;">**Applying the general rule blindly at exponent zero.** The task explicitly defines output one and gradient zero.</span>
* <span style="font-size: 14px;">**Using a fractional exponent with a negative base.** The result may leave the real-number domain.</span>
* <span style="font-size: 14px;">**Returning values in the wrong order.** The exponential output and gradient come first, followed by the power output and gradient.</span>

<span style="font-size: 14px;">Both operations follow the same reverse-mode pattern: compute the forward value, derive the local slope, and multiply that slope by the gradient arriving from the rest of the graph.</span>

---