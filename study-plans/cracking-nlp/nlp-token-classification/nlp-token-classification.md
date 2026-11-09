<span style="font-size: 14px;">Token classification is one of the fundamental tasks in natural language processing, where every token in a sequence receives its own label from a predefined set.</span>

## <span style="font-size: 16px;">What Is Token Classification?</span>

<span style="font-size: 14px;">Unlike sequence classification (which assigns one label to an entire input), token classification assigns a separate label to each individual token. Common applications include:</span>

* <span style="font-size: 14px;">**Named Entity Recognition (NER)**: labeling tokens as person, organization, location, or none</span>
* <span style="font-size: 14px;">**Part-of-Speech Tagging**: labeling tokens as noun, verb, adjective, etc.</span>
* <span style="font-size: 14px;">**Chunking**: identifying phrase boundaries (noun phrase, verb phrase)</span>

## <span style="font-size: 16px;">From Logits to Predictions</span>

<span style="font-size: 14px;">A token classification model produces a logit vector for each token. Given $N$ tokens and $K$ possible labels, the output is a matrix:</span>

$$
L \in \mathbb{R}^{N \times K}
$$

<span style="font-size: 14px;">Each row $L_i$ contains the raw scores for token $i$ across all $K$ labels. The predicted label is determined by the argmax operation:</span>

$$
\hat{y}_i = \arg\max_{j \in \{1, \dots, K\}} L_{i,j}
$$

<span style="font-size: 14px;">This selects the label index with the highest score. If two labels share the same maximum score, a common convention is to select the one with the smaller index.</span>

## <span style="font-size: 16px;">Evaluation with Accuracy</span>

<span style="font-size: 14px;">The simplest metric for token classification is per-token accuracy:</span>

$$
\text{accuracy} = \frac{\text{number of correctly predicted tokens}}{\text{total number of tokens}}
$$

<span style="font-size: 14px;">While accuracy is intuitive, it can be misleading when label distributions are imbalanced. For example, in NER, most tokens are labeled "O" (outside any entity), so a model predicting "O" for everything could achieve high accuracy. In practice, metrics like precision, recall, and F1-score per entity type are preferred for NER evaluation.</span>

## <span style="font-size: 16px;">Label Schemes</span>

<span style="font-size: 14px;">Token classification tasks often use structured label schemes to encode boundaries:</span>

* <span style="font-size: 14px;">**IOB2 (Inside-Outside-Beginning)**: "B-PER" marks the start of a person entity, "I-PER" continues it, "O" is outside</span>
* <span style="font-size: 14px;">**BILOU**: adds "L" (last token of entity) and "U" (unit, single-token entity) for finer boundary information</span>

<span style="font-size: 14px;">These schemes help distinguish adjacent entities of the same type and are critical for correctly extracting entity spans from token-level predictions.</span>

## <span style="font-size: 16px;">Key Takeaways</span>

* <span style="font-size: 14px;">Token classification predicts one label per token, not per sequence</span>
* <span style="font-size: 14px;">The argmax of the logit vector determines the predicted label for each token</span>
* <span style="font-size: 14px;">Accuracy measures the fraction of tokens predicted correctly</span>
* <span style="font-size: 14px;">Structured label schemes like IOB2 encode entity boundaries in the labels themselves</span>