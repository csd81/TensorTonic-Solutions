# <span style="font-size: 20px;">Trace a Reachable Expression Graph</span>

<span style="font-size: 14px;">A computation graph may contain nodes that have nothing to do with a particular output. Tracing the reachable graph means starting from one output and following parent links backward to find exactly the nodes that can influence it.</span>

---

## <span style="font-size: 16px;">Reachability means dependency</span>

<span style="font-size: 14px;">The requested output is always reachable from itself. Its parents are also reachable, as are their parents, continuing until the traversal reaches leaves with no parents. This complete set is the output's dependency closure.</span>

<span style="font-size: 14px;">A disconnected leaf or branch is excluded because there is no parent path from the output back to it. Its stored value may be valid, but it cannot affect the chosen result.</span>

---

## <span style="font-size: 16px;">Trace backward through parent links</span>

<span style="font-size: 14px;">The records store edges from parents into children, but discovery begins at the child chosen as the output. A stack or recursive traversal can follow the stored parent identifiers backward.</span>

<span style="font-size: 14px;">Maintain a set of identifiers already visited. When a node is encountered for the first time, mark it reachable and schedule its parents. If another path reaches the same node later, the visited set prevents unnecessary repeated traversal.</span>

<span style="font-size: 14px;">Shared parents are common in computation graphs. The visited set deduplicates reachable node identifiers, but it does not erase graph edges.</span>

---

## <span style="font-size: 16px;">Discovery order is not return order</span>

<span style="font-size: 14px;">A backward traversal naturally discovers the output before its ancestors. The required reachable identifiers use a different order: their original order in the supplied node list.</span>

<span style="font-size: 14px;">A reliable approach separates discovery from formatting:</span>

* <span style="font-size: 14px;">First, traverse backward and build the set of reachable identifiers.</span>
* <span style="font-size: 14px;">Then, scan the original node list and keep only records whose identifiers belong to that set.</span>

<span style="font-size: 14px;">This produces deterministic output without depending on stack order or recursion order.</span>

---

## <span style="font-size: 16px;">Constructing the ordered edges</span>

<span style="font-size: 14px;">Each returned edge points from a parent to its child:</span>

$$
\text{edge} = [\text{parent identifier},\ \text{child identifier}]
$$

<span style="font-size: 14px;">Scan reachable child records in their original input order. For each child, scan its parent list in stored order and emit one edge for every parent position.</span>

<span style="font-size: 14px;">If the same parent occupies both operand positions, emit two equal edges. The two positions represent two uses of that parent in the operation, so removing the duplicate would lose information from the stored graph.</span>

---

## <span style="font-size: 16px;">A small graph example</span>

<span style="font-size: 14px;">Suppose node $e$ multiplies leaves $a$ and $b$, node $d$ adds $e$ and $c$, and output $L$ multiplies $d$ and $f$. An additional leaf named $u$ is disconnected.</span>

$$
e = ab
$$

$$
d = e + c
$$

$$
L = df
$$

<span style="font-size: 14px;">Tracing backward from $L$ reaches $d$ and $f$, then $e$ and $c$, then $a$ and $b$. The disconnected leaf $u$ is never encountered.</span>

<span style="font-size: 14px;">When the original list places the leaves before the operations, the reachable identifiers are returned as $a$, $b$, $c$, $f$, $e$, $d$, and $L$. The edges are emitted child by child, preserving the stored parent order for each operation.</span>

---

## <span style="font-size: 16px;">Leaf outputs and shared parents</span>

<span style="font-size: 14px;">If the requested output is a leaf, only that identifier is reachable and there are no edges. If a parent contributes to several later nodes, it appears once in the reachable identifier list but once per connection in the edge list.</span>

<span style="font-size: 14px;">These cases show why reachable nodes and reachable edges are related but different outputs. One describes membership in the dependency closure, while the other describes every contributing connection.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Following children instead of parents.** Starting at the output requires moving backward through its stored parent identifiers.</span>
* <span style="font-size: 14px;">**Returning traversal order.** Stack order is an implementation detail; reachable identifiers must follow the original node-list order.</span>
* <span style="font-size: 14px;">**Removing repeated edges.** Two identical parent positions represent two contributions and must remain visible.</span>
* <span style="font-size: 14px;">**Including disconnected records.** A node belongs only when a backward path from the output reaches it.</span>
* <span style="font-size: 14px;">**Mutating parent lists.** Reordering records during traversal changes the supplied graph and can silently alter the required edge order.</span>

<span style="font-size: 14px;">The central technique is to discover reachability backward, then use the original graph order to produce stable node and edge outputs.</span>

---