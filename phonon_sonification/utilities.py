def format_duration_for_strauss(duration_seconds: float) -> str:
    """Convert duration in seconds to STRAUSS Score format "Xm Ys" """
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    return f"{minutes}m {seconds}s"
