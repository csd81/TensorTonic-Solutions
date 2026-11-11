def validate_records(records: list, schema: list) -> list:
    """
    Returns a list of result dictionaries.
    """
    results = []

    for idx, record in enumerate(records):
        errors = []

        for rule in schema:
            col = rule["column"]
            expected_type = rule["type"]
            nullable = rule["nullable"]

            # 1. Presence check
            if col not in record:
                errors.append(f"{col}: missing")
                continue

            val = record[col]

            # 2. Nullability check
            if val is None:
                if not nullable:
                    errors.append(f"{col}: null")
                continue

            # 3. Type check (Note: bool is a subclass of int in Python, so exclude it first)
            actual_type = type(val).__name__
            is_valid_type = False

            if isinstance(val, bool):
                is_valid_type = False
            elif expected_type == "int":
                is_valid_type = isinstance(val, int)
            elif expected_type == "float":
                is_valid_type = isinstance(val, (int, float))
            elif expected_type == "str":
                is_valid_type = isinstance(val, str)

            if not is_valid_type:
                errors.append(f"{col}: expected {expected_type}, got {actual_type}")
                continue

            # 4. Inclusive range check (applies only after a valid numeric type check)
            if expected_type in ("int", "float"):
                min_val = rule.get("min")
                max_val = rule.get("max")

                if (min_val is not None and val < min_val) or (
                    max_val is not None and val > max_val
                ):
                    errors.append(f"{col}: out of range")

        results.append(
            {"record_index": idx, "is_valid": len(errors) == 0, "errors": errors}
        )

    return results
