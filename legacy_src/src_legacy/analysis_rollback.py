def apply_rollback(distance, percent_loss):
    """
    Apply rollback to driving distance.
    Example: percent_loss = 0.05 means 5% reduction.
    """

    return distance * (1 - percent_loss)
