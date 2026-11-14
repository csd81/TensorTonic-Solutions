def create_nsp_pairs(documents: list, pair_specs: list) -> list:
  """
    Returns sentence_a, sentence_b, and is_next dictionaries in a list.
    """
  result = []
  for spec in pair_specs:
    doc_a_idx = spec["doc_a"]
    sent_a_idx = spec["sent_a"]
    doc_b_idx = spec["doc_b"]
    sent_b_idx = spec["sent_b"]

    sentence_a = documents[doc_a_idx][sent_a_idx]
    sentence_b = documents[doc_b_idx][sent_b_idx]

    # Check if they are from the same document and B immediately follows A
    is_next = (
        1
        if (doc_a_idx == doc_b_idx and sent_b_idx == sent_a_idx + 1)
        else 0
    )

    result.append(
        {"sentence_a": sentence_a, "sentence_b": sentence_b, "is_next": is_next}
    )

  return result
