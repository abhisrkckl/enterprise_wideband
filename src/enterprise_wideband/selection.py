import numpy as np


def by_groups(flags):
    flagvals = np.unique(flags["group"])
    return {val: flags["group"] == val for val in flagvals}