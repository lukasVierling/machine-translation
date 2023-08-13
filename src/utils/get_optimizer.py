import torch

def get_optimizer(model, optimizer, learning_rate, 
                  adam_beta1=0.9, adam_beta2=0.999,
                  weight_decay=0, momentum=0, path=None
                  ):
    """
    Utility function to get an optimizer for a given model and optimizer name.

    Args:
        optimizer (str): Name of the optimizer to use.
        model (nn.Module): Model to optimize.
        learning_rate (float): Learning rate for the optimizer.
        adam_beta1 (float, optional): Beta1 for Adam optimizer. Defaults to 0.9.
        adam_beta2 (float, optional): Beta2 for Adam optimizer. Defaults to 0.999.
        weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.
        momentum (float, optional): Momentum for SGD and RMSprop optimizer. Defaults to 0.

    Returns:
        torch.optim.Optimizer: Optimizer for the given model.
    """
    if optimizer == 'adam':
        '''
        The Adam optimizer maintains a learning rate for each parameter in a neural network. 
        It computes individual adaptive learning rates for each parameter by incorporating both the first 
        and second moments of the gradients. The first moment is the average of the gradients, 
        and the second moment is the average of the squared gradients.
        '''
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2), weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        '''
        The RMSProp optimizer keeps track of the magnitude of past gradients 
        by maintaining the moving average of squared gradients. 
        This allows it to adaptively adjust the learning rate for each parameter 
        based on the history of gradients encountered during training.
        '''
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer == 'adagrad':
        '''
        The key idea behind Adagrad is that it adapts the learning rate for each parameter by giving more weight
        to infrequent and important features. 
        Parameters that receive large gradients will have their learning rate scaled down, 
        while parameters with small gradients will have their learning rate scaled up.
        '''
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        '''
        you know this one
        '''
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'adadelta':
        '''
        Adadelta utilizes a running average of squared parameter updates (E[Δθ^2]). 
        This running average acts as a momentum term that adapts the learning rate based on 
        the historical information of parameter updates. 
        This allows Adadelta to continue learning even when the gradients become very small.
        Combines advatages of RMSPropo and Adagrad.
        '''
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    return optimizer

