import attr
import enum
from typing import List


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
        self.num_regions = len(self.regions)

    def unpack(self, y):
        """
        Unpacks the vector into the S, I and R variables in that order
        """
        n = self.num_regions
        return np.split(y, [(i+1)*n for i in range(NUM_STATUSES-1)])

    def pack(self, S, I, R):
        """
        Packs variables S, I and R into a vector by stacking them
        in that order
        """
        return np.concatenate((S, I, R), axis=0)
