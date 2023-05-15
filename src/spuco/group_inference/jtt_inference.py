from typing import Dict, List, Tuple
from spuco.group_inference import BaseGroupInference 

class JTTInference(BaseGroupInference):
    def __init__(
        self,
        predictions: List[int],
        labels: List[int],
    ):
        self.predictions = predictions
        self.labels = labels

    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        # Tuple as key to be consistent (0, 0) - correct set, (0, 1) - error set
        group_partition = {(0, 0): [], (0, 1): []} 

        for i, label in enumerate(self.labels):
            if label == self.predictions[i]:
                group_partition[(0, 0)].append(i)
            else:
                group_partition[(0, 1)].append(i)

        return group_partition

