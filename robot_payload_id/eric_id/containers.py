import itertools


def take_first(iterable):
    """
    Robustly gets the first item from an iterable and returns it.
    You should always use this instead of `next(iter(...))`; e.g. instead of

        my_first = next(iter(container))

    you should instead do:

        my_first = take_first(container)
    """
    (first,) = itertools.islice(iterable, 1)
    return first


def dict_items_zip(*items):
    """
    Provides `zip()`-like functionality for the items of a list of
    dictionaries. This requires that all dictionaries have the same keys
    (though possibly in a different order).

    Returns:
        Iterable[key, values], where ``values`` is a tuple of the value from
        each dictionary.
    """
    if len(items) == 0:
        # Return an empty iterator.
        return
    first = items[0]
    assert isinstance(first, dict)
    check_keys = set(first.keys())
    for item in items[1:]:
        assert isinstance(item, dict)
        assert set(item.keys()) == check_keys
    for k in first.keys():
        values = tuple(item[k] for item in items)
        yield k, values
