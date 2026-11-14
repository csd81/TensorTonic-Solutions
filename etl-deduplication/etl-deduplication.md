## The Problem: The Same Data Appears Multiple Times

Duplicate records are one of the most common and persistent data quality issues in data pipelines. They arise from many sources:

**At-Least-Once Delivery**: Message queues (Kafka, RabbitMQ, SQS) typically guarantee at-least-once delivery, meaning the same message may be delivered multiple times if acknowledgments fail.

**API Retries**: When an API call times out, clients often retry. If the original request actually succeeded, you now have the same data submitted twice.

**Merge Operations**: Combining data from multiple sources (e.g., merging acquisitions, consolidating regional databases) often introduces overlapping records.

**Backfill Operations**: Re-processing historical data may load records that already exist in the destination.

**User Actions**: Users may accidentally click a submit button twice or refresh a page that re-sends a form.

If duplicates are not handled, they cause severe downstream problems:
- Analytics overcounts events, leading to wrong business decisions
- ML models train on inflated data, skewing predictions
- Financial reports show incorrect totals
- Customer communications may be sent multiple times

**Deduplication** is the process of identifying and removing duplicate records, keeping only one representative record per unique entity.

---

## Defining Uniqueness: Key Columns

To deduplicate, you must first define what makes two records "the same." This is done through **key columns** (also called deduplication keys or natural keys).

Two records are duplicates if they have identical values in ALL key columns.

**Example with single key:**
- Key column: order_id
- Records with the same order_id are duplicates

**Example with composite key:**
- Key columns: (user_id, product_id, date)
- Records with the same user, product, AND date are duplicates

Choosing the right key columns requires domain knowledge:
- Too narrow (few columns): Unrelated records may be incorrectly flagged as duplicates
- Too wide (many columns): True duplicates may not be caught because they differ in irrelevant fields

---

## Deduplication Strategies

When duplicates exist, you must choose which record to keep. Different strategies serve different use cases:

### Strategy 1: "first"

Keep the first occurrence of each unique key (by position in the input).

**Use case**: When the first arrival is most reliable, such as:
- The original event before any retries
- The authoritative source when merging multiple feeds
- Preserving historical state when later records are corrections

**Algorithm**:
- For each record, extract the key
- If this key has not been seen before, keep the record
- If this key has been seen, discard the record

### Strategy 2: "last"

Keep the last occurrence of each unique key (by position in the input).

**Use case**: When the most recent arrival is most current, such as:
- Updates that supersede previous values
- Corrections that fix earlier mistakes
- "Upsert" semantics where later data is the truth

**Algorithm**:
- For each record, extract the key
- Always store (or replace) the record for this key
- After processing all records, output the stored records

### Strategy 3: "most_complete"

Keep the record with the fewest null values across ALL fields (not just key columns).

**Use case**: When different duplicates may have different missing fields, such as:
- Merging partial records from different sources
- Data enrichment where some duplicates have more information
- Handling records with optional fields that may or may not be populated

**Algorithm**:
- For each record, extract the key
- Count nulls in the entire record (all columns)
- Keep the record with the lowest null count
- Break ties by keeping the first occurrence

---

## Preserving Output Order

Regardless of which deduplication strategy is used, the output must preserve the **first-appearance order** of unique keys.

This means:
- If key A first appears at input index 5
- And key B first appears at input index 2
- Then in the output, B record appears before A record

This requirement ensures deterministic, reproducible output and maintains intuitive ordering that reflects when each unique entity was first seen.

---

## The Deduplication Algorithm

**Step 1: Initialize Tracking Structures**
- A dictionary mapping key tuples to lists of records with that key
- A list tracking the order in which keys first appeared

**Step 2: Group Records by Key**
- For each record in input order:
  - Extract the key tuple from key columns
  - If this is a new key, add it to the order list
  - Append the record to the list for this key

**Step 3: Select One Record Per Key**
- For each key (in first-appearance order):
  - Apply the strategy to its list of records
  - "first": take the first record in the list
  - "last": take the last record in the list
  - "most_complete": find the record with fewest nulls (first if tied)

**Step 4: Build Output**
- Collect selected records in first-appearance key order
- Return the deduplicated list

---

## Computing Null Counts

For the "most_complete" strategy, you need to count null values:

$$
\text{null\_count}(record) = \sum_{field \in record} \mathbf{1}[\text{value is None}]
$$

Count nulls across ALL fields in the record, not just key columns. This ensures you select the most informative record overall.

When multiple records have the same null count, the tie-breaker is position: keep the record that appeared first in the input.

---

## A Detailed Worked Example

**Input Records:**
- Record 0: {user_id: "A", product_id: 1, price: 10.0, notes: None}
- Record 1: {user_id: "B", product_id: 2, price: 20.0, notes: "rush"}
- Record 2: {user_id: "A", product_id: 1, price: 15.0, notes: "gift"}
- Record 3: {user_id: "A", product_id: 1, price: None, notes: "updated"}
- Record 4: {user_id: "B", product_id: 2, price: 25.0, notes: None}

**Key Columns:** [user_id, product_id]

**Grouping by Key:**
- Key ("A", 1): [Record 0, Record 2, Record 3]
- Key ("B", 2): [Record 1, Record 4]

**First-appearance order:** [("A", 1), ("B", 2)]

**Strategy: "first"**
- Key ("A", 1): select Record 0
- Key ("B", 2): select Record 1
- Output: [Record 0, Record 1]

**Strategy: "last"**
- Key ("A", 1): select Record 3
- Key ("B", 2): select Record 4
- Output: [Record 3, Record 4]

**Strategy: "most_complete"**
- Key ("A", 1): 
  - Record 0: nulls = 1 (notes is None)
  - Record 2: nulls = 0
  - Record 3: nulls = 1 (price is None)
  - Winner: Record 2 (0 nulls)
- Key ("B", 2):
  - Record 1: nulls = 0
  - Record 4: nulls = 1 (notes is None)
  - Winner: Record 1 (0 nulls)
- Output: [Record 2, Record 1]

Note: Output order follows first-appearance of keys (A,1 before B,2), even though the selected records may differ.

---

## Handling Composite Keys

When multiple columns form the key, the key is a tuple of values:

```python
key = tuple(record[col] for col in key_columns)
```

**Example:**
- Key columns: ["user_id", "date"]
- Record: {user_id: "alice", date: "2024-01-15", amount: 100}
- Key tuple: ("alice", "2024-01-15")

Two records are duplicates if their key tuples are equal (element-wise comparison).

**Important**: None values in key columns are valid and participate in comparison. Two records both having user_id=None are considered duplicates if that is the only key column.

---

## Edge Cases

**No Duplicates**: If every record has a unique key, all records are returned in original order.

**All Duplicates**: If all records share the same key, exactly one record is returned (determined by strategy).

**Empty Input**: Return an empty list.

**Single Record**: Return that record.

**Nulls in Key Columns**: Null values are compared using standard equality. Two nulls are considered equal.

---

## Computational Complexity

Deduplication scales linearly with input size:

$$
O(n \cdot k)
$$

where $n$ is the number of records and $k$ is the number of key columns.

For "most_complete" strategy, add $O(n \cdot c)$ where $c$ is the total number of columns (for null counting).

Hash-based key lookup ensures constant-time duplicate detection.

---

## Where Deduplication Shows Up

**Data Warehousing**: ETL pipelines deduplicate before loading to maintain data integrity in fact and dimension tables.

**Event Processing**: Stream processors deduplicate events to ensure exactly-once semantics.

**Data Integration**: When merging data from multiple sources, deduplication resolves overlapping records.

**CRM Systems**: Customer records from different touchpoints are deduplicated to create a unified customer view.

**Log Aggregation**: Duplicate log entries from retries or multiple servers are consolidated for accurate analysis.

**Machine Learning**: Training data is deduplicated to prevent models from overfitting to repeated examples.