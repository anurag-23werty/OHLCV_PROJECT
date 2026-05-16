
def validate_ohlcv(candle):

    errors = []

    # Rule 1: High should be >= Open
    if candle["high"] < candle["open"]:
        errors.append("High cannot be less than Open")

    # Rule 2: High should be >= Close
    if candle["high"] < candle["close"]:
        errors.append("High cannot be less than Close")

    # Rule 3: Low should be <= Open
    if candle["low"] > candle["open"]:
        errors.append("Low cannot be greater than Open")

    # Rule 4: Low should be <= Close
    if candle["low"] > candle["close"]:
        errors.append("Low cannot be greater than Close")

    # Rule 5: Volume cannot be negative
    if candle["volume"] < 0:
        errors.append("Volume cannot be negative")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }