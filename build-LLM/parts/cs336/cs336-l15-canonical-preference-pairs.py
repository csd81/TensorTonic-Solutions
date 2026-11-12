def canonical_preference_pairs(records: list) -> dict:
    """
    Returns a dict with pairs: a list of prompt_id, winner_id, loser_id dictionaries.
    """
    pairs = []

    for record in records:
        prompt_id = record["prompt_id"]
        ranked_candidates = sorted(record["candidates"], key=lambda c: c["rank"])
        num_candidates = len(ranked_candidates)

        for i in range(num_candidates):
            winner_id = ranked_candidates[i]["id"]
            for j in range(i + 1, num_candidates):
                loser_id = ranked_candidates[j]["id"]
                pairs.append(
                    {
                        "prompt_id": prompt_id,
                        "winner_id": winner_id,
                        "loser_id": loser_id,
                    }
                )

    return {"pairs": pairs}
