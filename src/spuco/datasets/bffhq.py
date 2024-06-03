from tqdm import tqdm 
import os

from spuco.datasets import BaseSpuCoCompatibleDataset

class bFFHQ(BaseSpuCoCompatibleDataset):
    def __init__(
        self,
        root: str,
        split: str,
        verbose: bool = False,
        transform=None,
    ):
        super().__init__()
        
        self.verbose =  verbose 
        self.data = []
        self._spurious = []
        self._labels = []
        self.transform = transform 
        
        # Process valid split
        if split == "val":
            split == "valid"
            
        # Load dataset
        if split == "train":
            if self.verbose:
                print("Loading dataset")
            for folder in ["align", "conflict"]:
                for label in ["0", "1"]:
                    dir_path = os.path.join(root, "0.5pct", folder, label)
                    for filename in tqdm(os.listdir(dir_path), desc=f"Loading class {label} bias-{folder}ing examples", disable=not self.verbose):
                        filepath = os.path.join(dir_path, filename)
                        if not os.path.isfile(filepath):
                            continue
                        self.data.append(filepath)
                        self._labels.append(int(label))
                        if folder == "align":
                            self._spurious.append(int(label))
                        else:
                            self._spurious.append(1 - int(label))                 
        elif split != "valid" or split != "test":
            dir_path = os.path.join(root, split)
            for filename in tqdm(os.listdir(dir_path), desc="Loading dataset", disable=not self.verbose):
                filepath = os.path.join(dir_path, filename)
                if not os.path.isfile(filepath):
                    continue 
                
                split_filename = filename.split("_")
                label = int(split_filename[1])
                spurious = int(split_filename[2].split(".")[0])

                self.data.append(filepath)
                self._labels.append(label)
                self._spurious.append(spurious)
        else:
            raise ValueError("Invalid split: choose one of train, test, val")
        
        self._group_partition = {(0,0): [], (0,1): [], (1,0): [], (1,1): []}    
        for i in tqdm(range(len(self.data)), desc="Initializing group partition", disable=not self.verbose):
            self._group_partition[(self._labels[i], self._spurious[i])].append(i)
        self._group_weights = {}
        for key in self.group_partition.keys():
            self._group_weights[key] = len(self.group_partition[key]) / len(self.data)
            
        self._num_classes = 2 
        
        if self.verbose:
            print("Done!")