from typing import Dict, List, Tuple

from spuco.group_inference import BaseGroupInference


class JTTInference(BaseGroupInference):
    """
    Just Train Twice Inference: https://arxiv.org/abs/2107.09044
    """
    def __init__(
        self,
        predictions: List[int],
        class_labels: List[int],
    ):
        """
        Initializes JTTInference.

        :param predictions: List of predicted labels.
        :param class_labels: List of true class labels.
        """
        super().__init__()
        self.predictions = predictions
        self.class_labels = class_labels

    def infer_groups(self) -> Dict[Tuple[int, int], List[int]]:
        """
        Infers group partitions based on predictions and class labels.

        :return: Dictionary mapping group tuples to indices of examples belonging to each group.
        """
        group_partition = {(0, 0): [], (0, 1): []} 

        for i, label in enumerate(self.class_labels):
            if label == self.predictions[i]:
                group_partition[(0, 0)].append(i)
            else:
                group_partition[(0, 1)].append(i)

        return group_partition

