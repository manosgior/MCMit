def process_distribution(counts: dict[str, int]):
    new_counts = {}

    for key, value in counts.items():
        strings = key.split(' ')
        longer_string = strings[0] if len(strings[0]) > len(strings[1]) else strings[1]
        new_counts[longer_string] = new_counts.get(longer_string, 0) + value
       
    return new_counts