from torch.utils.data import Dataset 

class IndexDatasetWrapper(Dataset):
    """
    Wrapper class that takes a Dataset and returns a dataset 
    where the indices of the data are returned as well.
    """
    def __init__(
        self,
        dataset: Dataset,
    ):
        """
        Initialize an instance of IndexDatasetWrapper.

        :param dataset: The original dataset.
        :type dataset: Dataset
        """
        
        self.dataset = dataset
        self.idx = range(len(self.dataset))
            
    def __getitem__(self, index):
        """
        Get an item from the dataset.

        :param index: The index of the item to retrieve.
        :type index: int
        :return: A tuple containing the input data, the label, and the index.
        :rtype: Tuple[Any, int, int]
        """
        index = self.idx[index]
        return (self.dataset.__getitem__(index)[0], self.dataset.__getitem__(index)[1], index)
    
    def __len__(self):
        """
        Get the length of the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.idx)
        