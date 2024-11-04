import torch.nn as nn

class My_Model(nn.Module):
    '''
    A simple feed-forward neural network for regression tasks.
    
    Architecture:
        - Input layer: input_dim neurons
        - First hidden layer: 16 neurons with ReLU activation
        - Second hidden layer: 8 neurons with ReLU activation
        - Output layer: 1 neuron (for regression prediction)
    '''
    def __init__(self, input_dim):
        '''
        Initialize the neural network model.
        
        Args:
            input_dim (int): Number of input features
        '''
        super(My_Model, self).__init__()
        # Define the sequential model with three linear layers and ReLU activations
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),  # First layer: input_dim -> 16 neurons
            nn.ReLU(),                 # ReLU activation function
            nn.Linear(16, 8),          # Second layer: 16 -> 8 neurons
            nn.ReLU(),                 # ReLU activation function
            nn.Linear(8, 1)            # Output layer: 8 -> 1 neuron (regression output)
        )

    def forward(self, x):
        '''
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Model predictions of shape (batch_size,)
        '''
        # Pass input through layers and squeeze the output to match target dimensions
        return self.layers(x).squeeze(1)
