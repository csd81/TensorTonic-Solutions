# <span style="font-size: 20px;">Build a Scalar Expression Graph</span>

<span style="font-size: 14px;">A computation graph records a calculation as a sequence of nodes. Leaf nodes supply the starting numbers, while operation nodes combine earlier values. Building the graph means evaluating each operation in order and preserving enough history to explain where every result came from.</span>

---

## <span style="font-size: 16px;">Leaves and operation nodes</span>

<span style="font-size: 14px;">Every returned node has the same five-field structure, but leaves and operations use it differently.</span>

* <span style="font-size: 14px;">A **leaf node** stores an identifier and scalar data. Its operation is empty because the graph did not compute it, and its parent list is empty.</span>
* <span style="font-size: 14px;">An **operation node** stores a newly computed scalar value, the addition or multiplication symbol, and the identifiers of its two parents.</span>
* <span style="font-size: 14px;">Every gradient starts at zero because graph construction is a forward-pass task.</span>

<span style="font-size: 14px;">This problem stores parent identifiers inside operation nodes. That is the required representation here, even though a different problem may retain parent objects directly.</span>

---

## <span style="font-size: 16px;">Why creation order matters</span>

<span style="font-size: 14px;">Each operation is guaranteed to refer only to nodes created earlier. This makes the supplied order a valid forward evaluation order. When an operation is reached, both parent values are already available.</span>

<span style="font-size: 14px;">A lookup from identifier to node makes parent retrieval direct. After creating a node, add it to both the ordered result and the lookup. The ordered result preserves the graph's construction history, while the lookup provides the data needed by later operations.</span>

---

## <span style="font-size: 16px;">Evaluating the two supported operations</span>

<span style="font-size: 14px;">For addition, combine the parent values as:</span>

$$
z = x + y
$$

<span style="font-size: 14px;">For multiplication, combine them as:</span>

$$
z = xy
$$

<span style="font-size: 14px;">The numeric result goes into the data field. The operation symbol and ordered parent identifiers preserve the topology. Keeping value and topology separate is important because two nodes can hold the same number while having completely different histories.</span>

---

## <span style="font-size: 16px;">Following a complete expression</span>

<span style="font-size: 14px;">Consider four leaves with values $2$, $-3$, $10$, and $-2$. Give them the identifiers $a$, $b$, $c$, and $f$. Three operations then build the expression:</span>

$$
e = ab = 2(-3) = -6
$$

$$
d = e + c = -6 + 10 = 4
$$

$$
L = df = 4(-2) = -8
$$

<span style="font-size: 14px;">The returned nodes appear in creation order: the four leaves first, followed by $e$, $d$, and $L$. The final node identifier is $L$ because it was created by the last operation.</span>

<span style="font-size: 14px;">The parent identifiers record the structure. Node $e$ points to $a$ and $b$, node $d$ points to $e$ and $c$, and node $L$ points to $d$ and $f$.</span>

---

## <span style="font-size: 16px;">The case with no operations</span>

<span style="font-size: 14px;">A graph can consist only of leaves. When the operation sequence is empty, no new node is created, so the final node is the last leaf. The returned list still contains normalized five-field records for every leaf.</span>

<span style="font-size: 14px;">This case is easy to overlook because most examples end with an operation. Handling it directly keeps the return contract consistent for the smallest valid graph.</span>

---

## <span style="font-size: 16px;">Implementation order</span>

* <span style="font-size: 14px;">Create fresh node records for all leaves, preserving their supplied order.</span>
* <span style="font-size: 14px;">Initialize every leaf gradient to zero and leave its operation and parents empty.</span>
* <span style="font-size: 14px;">Record each new node in an identifier lookup.</span>
* <span style="font-size: 14px;">Process operations in order, retrieve the two earlier parent values, and compute the requested sum or product.</span>
* <span style="font-size: 14px;">Append each operation node with its ordered parent identifiers.</span>
* <span style="font-size: 14px;">Return the complete node list together with the last node's identifier.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Confusing parent objects with parent identifiers.** This graph contract stores the two identifiers, not copies or references to the full records.</span>
* <span style="font-size: 14px;">**Losing creation order.** Returning lookup values can produce the wrong ordering even when every node is present.</span>
* <span style="font-size: 14px;">**Reversing the parents.** The left identifier must remain first and the right identifier second.</span>
* <span style="font-size: 14px;">**Mutating the inputs.** The returned nodes must be fresh records, leaving the supplied leaves and operations unchanged.</span>
* <span style="font-size: 14px;">**Assuming an operation always exists.** With no operations, the last leaf is the final node.</span>

<span style="font-size: 14px;">A correctly built graph preserves three things at once: the scalar result at every stage, the ordered dependency structure, and the exact sequence in which the nodes were created.</span>

---