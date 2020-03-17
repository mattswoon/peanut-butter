import attr
import enum
from typing import List

import numpy as np


@attr.s
class Region:
    code = attr.ib(type=str)


class Status(enum.Enum):
    susceptible = enum.auto()
    infected = enum.auto()
    recovered = enum.auto()


NUM_STATUSES = 3


@attr.s
class Indexer:
    regions = attr.ib(type=List[Region])

    def __attrs_post_init__(self):
        self.regions = sorted(self.regions)
        idx = 0
        fwd = {}
        bwd = {}
        for r in self.regions:
            fwd[r] = idx
            bwd[idx] = r
            idx += 1
        self._index = fwd
        self._region = bwd
        self._n = idx

    def vector(self, map: Dict[Region, float]):
        v = np.zeros(self._n)
        for r, val in map.items():
            v[self._index[r]] = val
        return v

    def matrix(self, entries: List[Dict[str, float]]):
        pass

    def unpack(self, y):
        """
        Unpacks the vector into the S, I and R variables in that order
        """
        n = self._n
        return np.split(y, [(i+1)*n for i in range(NUM_STATUSES-1)])

    def pack(self, S, I, R):
        """
        Packs variables S, I and R into a vector by stacking them
        in that order
        """
        return np.concatenate((S, I, R), axis=0)
