import heapq
import math

import numpy as np
from tqdm import tqdm


class FacilityLocation:
    def __init__(self, D, V, alpha=1.):
        '''
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        '''
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset) > 0:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            return self.norm * math.log(1 + self.f_norm * np.maximum(self.curMax, self.D[:, ndx]).sum()) - self.curVal
        else:
            return self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum()) - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy(F, V, B, verbose=False):
    """
    Args
    - F: FacilityMaximization Object
    - V: list of indices of columns of Similarity Matrix
    - B: Budget of subset (int)
    """
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)
    [_heappush_max(order, (F.inc(sset, index), index)) for index in V]

    if verbose:
        print("Submodular Maximization: Selecting subset...")
    with tqdm(total=B, disable=not verbose) as pbar:
        while order and len(sset) < B:
            el = _heappop_max(order)
            improv = F.inc(sset, el[1])

            if improv >= 0:
                if not order:
                    curVal = F.add(sset, el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                    pbar.update(1)
                else:
                    top = _heappop_max(order)
                    if improv >= top[0]:
                        curVal = F.add(sset, el[1])
                        sset.append(el[1])
                        vals.append(curVal)
                        pbar.update(1)
                    else:
                        _heappush_max(order, (improv, el[1]))
                    _heappush_max(order, top)
    return sset, vals