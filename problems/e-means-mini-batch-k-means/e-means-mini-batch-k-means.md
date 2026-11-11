## Why does Lloyd's algorithm need so many passes?

Lloyd's algorithm alternates two steps over the whole dataset. It assigns every point to its nearest centroid, then moves every centroid to the exact mean of its points. Each iteration lowers the k-means loss, the mean squared distance from each point to its centroid, but it costs a full pass over the data, and a centroid only moves once per pass. On a dataset of a billion embeddings that is a billion distance computations per centroid move, and runs of 20 or more passes are common before the loss settles.

Mini-batch methods move the centroids after every small batch of points. A batch of a few thousand points already says roughly where each centroid should go, so the centroids move hundreds or thousands of times per pass instead of once. The hard part is deciding how much to trust each batch, because a single batch is a noisy sample of the data.

## Three ways to keep the statistics

Every centroid can be written as a sum divided by a count, c = S / N. The update methods differ only in how they combine the old S and N with the batch values s and n.

Lloyd's algorithm replaces them, so S becomes s and N becomes n. Only the latest pass counts, which works when the batch is the whole dataset and fails badly on small batches, where each centroid would jump to the mean of a handful of points.

MacQueen's online k-means adds them, so S becomes S plus s and N becomes N plus n. Nothing is ever forgotten. Points assigned in the first batch, when the centroids were still far from good, keep their full weight for the rest of training, so early mistakes are only diluted slowly and the centroids end up moving very little.

e-means blends them with an exponential moving average, so S becomes (1 minus alpha) times S plus alpha times s, and N follows the same rule. Recent batches carry the most weight and older ones fade at rate alpha. Setting alpha to 1 with one batch that holds the whole dataset recovers Lloyd's algorithm exactly, which is why the second public example matches a Lloyd step.

## Why average the sum and the count separately?

A simpler idea is a moving average of the centroid itself, c becomes (1 minus alpha) times c plus alpha times the batch mean. That rule treats a batch where a cluster received one point the same as a batch where it received five hundred. Keeping S and N separately weights each batch by how many points it actually sent to the cluster. A cluster that received many points in the latest batch moves almost all the way to their mean, and a cluster that received one point barely moves.

The same bookkeeping explains what happens to a cluster that receives no points. Its s and n are zero, so S and N both shrink by the factor 1 minus alpha and their ratio, the centroid, stays fixed. The count keeps fading, though, so when the cluster does receive points later, those points dominate its next position. A count close to zero is also a useful sign of a dead cluster, one that no longer wins any points.

## Why is a large step safe here?

Gradient descent on the k-means loss can overshoot when its step is large, and the centroid can land somewhere that no point is near. The e-means update cannot do that. With alpha between 0 and 1, every centroid is a weighted average of points that were assigned to it, plus its starting position, so it always stays inside the region those points span. This is the reason the method stays stable with steps close to 1, and the paper uses this property to explain why large steps work in practice.

## What does the full method add?

The update in this problem is the core of the method. The paper adds three practical pieces around it. It picks the batch size and alpha from fitted scaling laws in n, k and the number of epochs, so nobody has to sweep them. It lowers alpha over training on a cosine schedule, much like a learning rate schedule. It also revives clusters whose counts have faded by splitting a large cluster in two. With these pieces, the authors report beating the loss of 25 Lloyd epochs in 1.4 to 2.6 epochs on embedding datasets, with GPU wall-clock speedups of 10 to 45 times, and clustering 10 billion DINO embeddings into 16 million clusters in 2.5 epochs.

## How it comes up in interviews

Clustering at scale comes up in retrieval and data curation interviews, for example building an inverted file index for vector search or deduplicating a web-scale training set. A common question is how to run k-means when the data does not fit in memory. A good answer says that Lloyd's algorithm needs a full pass per update. It then explains that mini-batch k-means updates per batch, and that the choice of how to merge old and new statistics decides whether it converges well. A candidate who can explain why an exponential moving average of the sum and the count beats both replacing and accumulating, and why a cluster missing from a batch keeps its position, shows they understand the method beyond the library call.