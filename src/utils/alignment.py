def uniform_alignment(source_len, target_len):
    """
    Monotonic alignment function for aligning the target sentence with
    the source sentence in a uniform way.

    Args:
        source_len (int): The length of the source sentence. (including SOS and EOS)
        target_len (int): The length of the target sentence. (including SOS and EOS)

    Returns:
        Callable[[int], int]: A function that takes the index of the source word and returns
            the index of the aligned target word.
    """

    I = source_len - 1
    J = target_len - 1
    k = float(J) / float(I)

    def uniform_alignment(i: int) -> int:   
        return int(round(k * i))
    
    return uniform_alignment

def stepwise_alignment(source_len, target_len):
    """
    Monotonic alignment function for aligning the target sentence with
    the source sentence in a stepwise way.

    Args:
        source_len (int): The length of the source sentence. (including SOS and EOS)
        target_len (int): The length of the target sentence. (including SOS and EOS)

    Returns:
        Callable[[int], int]: A function that takes the index of the source word and returns
            the index of the aligned target word.
    """

    I = source_len - 1
    J = target_len - 1
    k = float(J) / float(I)

    def stepwise_alignment(i: int) -> int:
        if i == I:
            return J
        return min(i, J)
    
    return stepwise_alignment



