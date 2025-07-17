def process_distribution_ghz(counts: dict[str, int]):
    new_counts = {}

    for key, value in counts.items():
        strings = key.split(' ')
        longer_string = strings[0] if len(strings[0]) > len(strings[1]) else strings[1]
        new_counts[longer_string] = new_counts.get(longer_string, 0) + value
       
    return new_counts


def process_distribution_teleportation(counts: dict[str, int]):
    new_counts = {}

    for key, value in counts.items():
        strings = key.split(' ')
        smallest_string = [str for str in strings if len(str) == 1][0]
        new_counts[smallest_string] = new_counts.get(smallest_string, 0) + value
       
    return new_counts


def process_distribution_long_range_cnot(counts: dict[str, int]):
    new_counts = {}

    for key, value in counts.items():
        strings = key.split(' ')
        correct_string = strings[0]
        new_counts[correct_string] = new_counts.get(correct_string, 0) + value
        
    return new_counts