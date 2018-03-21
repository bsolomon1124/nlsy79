"""Unique variable-length key generator.  Credit @Martijn Pieters"""

import base64
from itertools import filterfalse, islice
import math
import os

import numpy as np

FACTOR = math.log(256, 32)


def unique_everseen(iterable, key=None):
    # Itertools Recipes
    # https://docs.python.org/3/library/itertools.html#itertools-recipes
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def produce_amount_keys(amount_of_keys):
    """Unique key generation."""
    # Copied and pasted:
    # https://stackoverflow.com/a/48421303/7954504
    def gen_keys(_urandom=os.urandom, _encode=base64.b32encode,
                 _randint=np.random.randint):
        # (count / math.log(256, 32)), rounded up, gives us the number of bytes
        # needed to produce *at least* count encoded characters
        in_len = [None]*12 + [math.ceil(i / FACTOR) for i in range(12, 20)]
        while True:
            count = _randint(12, 20)
            yield _encode(_urandom(in_len[count]))[:count].decode('ascii')
    return list(islice(unique_everseen(gen_keys()), amount_of_keys))
