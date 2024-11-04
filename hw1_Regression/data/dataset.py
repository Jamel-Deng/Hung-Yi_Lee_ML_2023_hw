import torch
from torch.utils.data import Dataset

class COVID19Dataset(Dataset):
    '''
    A custom Dataset class for COVID-19 data handling.
    
    This dataset class inherits from torch.utils.data.Dataset and is designed
    to handle COVID-19 related features and targets.
    
    Attributes:
        x (torch.FloatTensor): Input features tensor
        y (torch.FloatTensor or None): Target values tensor. 
                                     If None, the dataset is used for prediction only.
    '''
    def __init__(self, x, y=None):
        '''
        Initialize the COVID19Dataset.
        
        Args:
            x (array-like): Feature data to be converted to torch.FloatTensor
            y (array-like, optional): Target data to be converted to torch.FloatTensor.
                                    Defaults to None for prediction mode.
        '''
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        '''
        Get a data sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            If y is None:
                torch.FloatTensor: Features for the idx-th sample
            else:
                tuple: (features, target) for the idx-th sample
        '''
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        '''
        Get the total number of samples in the dataset.
        
        Returns:
            int: Length of the dataset
        '''
        return len(self.x)