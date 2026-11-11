# <span style="font-size: 20px;">Topologically Sort a Computation DAG</span>

<span style="font-size: 14px;">A computation graph is a directed acyclic graph in which parent nodes supply values to child nodes. A topological order arranges the reachable nodes so every parent appears before every child that depends on it.</span>

---

## <span style="font-size: 16px;">Why autodiff needs an ordering</span>

<span style="font-size: 14px;">During the forward pass, a child cannot be evaluated until all of its parent values are available. Parent-before-child order guarantees that condition.</span>

<span style="font-size: 14px;">During the backward pass, the direction reverses. Children must send their gradients before parents propagate further, so reverse-mode autodiff traverses the topological order backward.</span>

$$
\text{forward: parents before children}
$$

$$
\text{backward: children before parents}
$$

---

## <span style="font-size: 16px;">Only the selected output matters</span>

<span style="font-size: 14px;">The supplied graph may contain valid nodes that do not contribute to the selected output. Starting from that output and repeatedly following parent identifiers finds its reachable subgraph.</span>

<span style="font-size: 14px;">Disconnected nodes are excluded before ordering. This keeps the result focused on the calculation that can affect the selected output and prevents unrelated branches from entering a later backward pass.</span>

---

## <span style="font-size: 16px;">The topological condition</span>

<span style="font-size: 14px;">For every reachable edge from parent $p$ to child $c$, their positions in the result must satisfy:</span>

$$
\operatorname{position}(p) < \operatorname{position}(c)
$$

<span style="font-size: 14px;">A simple chain has only one possible topological order. A branching graph can have several valid orders because independent nodes do not constrain one another.</span>

<span style="font-size: 14px;">This problem removes that ambiguity by using original node input order to resolve every valid tie.</span>

---

## <span style="font-size: 16px;">Ready nodes and deterministic ties</span>

<span style="font-size: 14px;">A node is ready when all of its reachable parents have already been placed in the result. Leaves are ready immediately because they have no parents.</span>

<span style="font-size: 14px;">Several nodes may be ready at the same time. Choose the one that appeared earliest in the original node list. After placing it, some children may become ready. Keeping the ready choices ordered by their original positions makes the final result deterministic.</span>

<span style="font-size: 14px;">This is the central idea behind a stable form of Kahn's topological sorting method. Parent counts enforce validity, while original positions enforce the required tie rule.</span>

---

## <span style="font-size: 16px;">A diamond-shaped example</span>

<span style="font-size: 14px;">Consider independent leaves $a$ and $b$. Node $c$ depends on $b$, node $d$ depends on $a$, and the output depends on both $c$ and $d$.</span>

$$
b \longrightarrow c \longrightarrow \text{output}
$$

$$
a \longrightarrow d \longrightarrow \text{output}
$$

<span style="font-size: 14px;">Both leaves are initially ready. If $a$ appears before $b$ in the input list, $a$ is selected first. After $a$ is placed, $d$ becomes ready, but $b$ still appears earlier than $d$ in the original list, so $b$ is selected next. The deterministic order becomes $a$, $b$, $c$, $d$, followed by the output.</span>

<span style="font-size: 14px;">A depth-first traversal could produce another valid parent-before-child order, but it would not necessarily obey this global input-order tie rule.</span>

---

## <span style="font-size: 16px;">Reachability and parent counts</span>

<span style="font-size: 14px;">The parent count for a reachable node records how many reachable prerequisites remain. Leaves begin at zero. Every time a parent is emitted, each reachable child loses one remaining prerequisite. A child joins the ready set exactly when that count reaches zero.</span>

<span style="font-size: 14px;">The graph is guaranteed to be acyclic, so every reachable node eventually becomes ready. If a cycle existed, some parent counts would never reach zero and the result would remain incomplete.</span>

---

## <span style="font-size: 16px;">Special cases</span>

<span style="font-size: 14px;">If the selected output is a leaf, the reachable subgraph contains only that leaf and the result has one identifier. If a disconnected node appears earlier than every reachable node, it is still excluded because reachability takes precedence over input order.</span>

<span style="font-size: 14px;">Parent-list order does not replace the tie rule. The original order of all supplied node records is the authority whenever several valid nodes are ready together.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Sorting every graph node.** Disconnected nodes must be removed before producing the reachable topological order.</span>
* <span style="font-size: 14px;">**Returning backward order.** The requested result is forward topological order; reverse mode will reverse it later.</span>
* <span style="font-size: 14px;">**Using traversal order as the tie rule.** Stack or recursion behavior may violate original input order.</span>
* <span style="font-size: 14px;">**Emitting a child too early.** Every reachable parent must already appear before the child becomes ready.</span>
* <span style="font-size: 14px;">**Letting disconnected nodes affect ties.** Only reachable ready nodes participate in ordering decisions.</span>

<span style="font-size: 14px;">A deterministic topological order combines two guarantees: graph dependencies decide which nodes are eligible, and original input positions decide among nodes that are equally eligible.</span>

---