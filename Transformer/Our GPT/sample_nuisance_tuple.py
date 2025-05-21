import random

def sample_tuple(ranges):
    """
    Sample a tuple of floats/ints where each element is drawn from its own (min, max) range.

    Args:
        ranges (list or tuple): A list/tuple of (min, max) pairs for each element.

    Returns:
        tuple: A tuple of sampled values.
    """
    return tuple(random.uniform(low, high) for (low, high) in ranges)



if __name__ == '__main__':
    sampled = sample_tuple([(-3.0, -1.0), (25, 100)])
    print(sampled)  # e.g., (0.64, 15.2, -3.0)
    print(10**sampled[0])
    print(10**sampled[1])