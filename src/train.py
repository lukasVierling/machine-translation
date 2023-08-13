import sys
sys.path.append('..')

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torcheval.metrics import Perplexity, MulticlassAccuracy
from torchsummary import summary
from tqdm import tqdm

from data.dataset import MTDataset
from src.models import Model

def train(config):
    """
    Train a neural machine translation model. Log the training metrics in TensorBoard.

    Args:
        config (dict): Configuration parameters for training the model.

    Returns:
        None
    """
    # Unpack configuration dictionary
    name = config['name']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    embedding_dim = config['embedding_dim']
    linSource_dim = config['linSource_dim']
    linTarget_dim = config['linTarget_dim']
    lin1_dim = config['lin1_dim']
    activation = config['activation']
    optimizer = config['optimizer']
    epochs = config['epochs']
    log_dir = config['log_dir']
    model_dir = config['model_dir']
    train_source = config['train_source']
    train_target = config['train_target']
    dev_source = config['dev_source']
    dev_target = config['dev_target']
    bpe_ops = config['bpe_ops']
    joined_bpe = config['joined_bpe']
    window_size = config['window_size']
    batch_alignment = config['batch_alignment']
    n_checkpoint = config['n_checkpoint']
    n_evaluate = config['n_evaluate']
    strategy = config['strategy']
    zero_bias = config['zero_bias']

    print("BPE ops: ", bpe_ops)
    print("Joined BPE: ", joined_bpe)
    # Create a summary writer
    writer = SummaryWriter(log_dir=log_dir + f"/{name}/events")

    #load train and dev data
    print("Loading data...")
    train_dataset = MTDataset(
        train_source, 
        train_target, 
        bpe_ops = bpe_ops, 
        joined_bpe=joined_bpe,
        window_size=window_size,
        batch_alignment=batch_alignment
        )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = MTDataset(
        dev_source, 
        dev_target, 
        bpe_ops = bpe_ops, 
        joined_bpe=joined_bpe,
        window_size=window_size,
        batch_alignment=batch_alignment
        )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    
    # Create a model
    print("Creating a model...")
    print(f"Initialization Strategy: {strategy}")
    print(f'Zero bias: {zero_bias}')
    if model_dir is None:
        model = Model(
            train_dataset.get_source_vocab_size(), 
            train_dataset.get_target_vocab_size(), 
            embedding_dim,
            linSource_dim, 
            linTarget_dim, 
            lin1_dim, 
            window_size,
            activation,
            strategy,
            zero_bias)
    else:
        # Load the model from the given path
        model = torch.load(model_dir)

    # Move the model to the given device
    device = torch.device(config['device'])
    print(f"Device: {device}")
    model.to(device)

    # Define a loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define an optimizer
    print(f"Optimizer: {optimizer}")
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(config['adam_beta1'], config['adam_beta2']))
        print("Betas: ", (config['adam_beta1'], config['adam_beta2'], "\n"))
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=config['weight_decay'])
        print("Weight decay: ", config['weight_decay'], "\n")
    elif optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=config['weight_decay'])
        print("Weight decay: ", config['weight_decay'], "\n")
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=config['momentum'], weight_decay=config['weight_decay'])
        print("Momentum: ", config['momentum'])
    elif optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=config['weight_decay'])
        print("Weight decay: ", config['weight_decay'], "\n")
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=config['momentum'], weight_decay=config['weight_decay'])

    #Summary of the model
    summary(model)
    print(f"Model strucutre: {model}")
    
    # initialize learning rate halving
    halving = config['learning_rate_halving']
    halving_patience = config['learning_rate_patience']
    halving_epsilon = config['learning_rate_performance_epsilon']
    previous_ppls = []


    # Train the model
    print(f"Training the model for {epochs} epochs...\n")
    for epoch in range(epochs):
        model.train()

        # initialize performance metrics
        loss_epoch = 0
        multiclass_accuracy = MulticlassAccuracy(device=device)
        # perplexity currently not working with mps device
        if config["device"] != "mps":
            perplexity = Perplexity(device=device)
        else:
            perplexity = None

        i = 0

        progress_bar = tqdm(iter(train_loader), desc='Training Epoch {}'.format(epoch))
        # Iterate over all batches
        for batch in progress_bar:
            i += 1
            # save model checkpoints if checkpoints_per_epoch is set
            if config['checkpoints_per_epoch'] > 0:
                if i % (len(train_loader) // config['checkpoints_per_epoch']) == 0:
                    torch.save(model, log_dir + f'/{name}/checkpoints/model_{epoch}_{i}.pt')
                    torch.save(optimizer, log_dir + f'/{name}/checkpoints/optimizer_{epoch}_{i}.pt')
            # Reset gradients
            optimizer.zero_grad()
            
            # Move data to the correct device
            target, source, label = batch
            target = target.to(device)
            source = source.to(device)
            label = label.to(device)

            # Forward pass
            prediction = model(target,source)
            # Calculate loss
            loss = criterion(prediction, label.long())
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            # Log loss
            loss_epoch += loss.item()

            # make prediction of shape (batch_size, vocab_size) to shape (batch_size, 1, vocab_size) for perplexity calculation
            if perplexity:
                curr_batch_size = prediction.shape[0]
                ppl_pred = prediction.view(curr_batch_size, 1, -1).to(device)
                ppl_label = label.view(curr_batch_size, 1).to(device)

                perplexity.update(ppl_pred, ppl_label)
            multiclass_accuracy.update(prediction,label)

        # Average loss and accuracy over all batches
        loss_epoch /= len(train_loader)
        print("Average epoch loss (train): ", loss_epoch)
        print("Accuracy (train): ", multiclass_accuracy.compute().item())

        if perplexity:
            print("Perplexity (train): ", perplexity.compute().item())
        
        # Log training metrics in tensorboard
        log(epoch, loss_epoch, perplexity, multiclass_accuracy, writer)

        # halve learning rate if performance on dev set does not improve
        if halving:
            _, ppl, _ = evaluate(model, validation_loader, epoch)
            previous_ppls.append(ppl)
            # remove oldest ppl if patience is reached
            if len(previous_ppls) > halving_patience:
                previous_ppls.pop(0)
            if len(previous_ppls) == halving_patience and all([previous_ppl - ppl < halving_epsilon for previous_ppl in previous_ppls]):
                learning_rate /= 2
                print(f"Learning rate halved to {learning_rate}")
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate
                    previous_ppls = []
        # save model checkpoints if checkpoints_per_epoch is not set
        if config['checkpoints_per_epoch'] < 1 & epoch % n_checkpoint == 0:
            torch.save(model, log_dir + f'/{name}/checkpoints/model_{epoch}.pt')
            torch.save(optimizer, log_dir + f'/{name}/checkpoints/optimizer_{epoch}_{i}.pt')
        # evaluate model on dev set
        if epoch % n_evaluate == 0:
            dev_loss, dev_ppl, def_acc = evaluate(model, validation_loader, epoch, writer, device)
            print("\n")
            print("Average loss (dev): ", dev_loss)
            print("Accuracy (dev): ", def_acc.item())
            print("Perplexity (dev): ", dev_ppl.item())
        print("\n")
    # save final model
    torch.save(model, log_dir + f'/{name}/checkpoints/model_{epochs}.pt')
    torch.save(optimizer, log_dir + f'/{name}/checkpoints/optimizer_{epoch}_{i}.pt')
    writer.close()

    print("End of training \n\n\n")
    

def log(epoch, loss, perplexity, multiclass_accuracy, writer):
    """
    Log training metrics for a given epoch.

    Args:
        epoch (int): Current epoch number.
        loss (float): Average loss value for the epoch.
        perplexity (Perplexity or None): Perplexity object for computing perplexity metric, or None if not applicable.
        multiclass_accuracy (MulticlassAccuracy): MulticlassAccuracy object for computing accuracy metric.
        writer (SummaryWriter): SummaryWriter object for logging metrics to TensorBoard.
    Returns:
        None
    """
    # Log training metrics
    writer.add_scalar('Loss/train', loss, epoch)
    if perplexity is not None:
        writer.add_scalar('Perplexity/train', perplexity.compute(), epoch)
    writer.add_scalar('Accuracy/train', multiclass_accuracy.compute(), epoch)


def evaluate(model, validation_loader, epoch, writer=None, device=None):
    """
    Evaluate the model on the validation dataset.

    Args:
        model: The trained model to evaluate.
        validation_loader: DataLoader for the validation dataset.
        epoch (int): Current epoch number.
        writer (SummaryWriter or None): SummaryWriter object for logging metrics to TensorBoard, or None if not provided.
        device (torch.device or None): Device to use for evaluation, or None if not provided.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - loss (float): Average loss value on the validation dataset.
            - perplexity (float or None): Perplexity metric on the validation dataset, or None if device is 'mps'.
            - multiclass_accuracy (float): Multiclass accuracy on the validation dataset.
    """
    # Define a loss function
    criterion = nn.CrossEntropyLoss()
    if device != torch.device('mps'):
        perplexity = Perplexity()

    multiclass_accuracy = MulticlassAccuracy(device=device)
    # Evaluate the model
    model.eval()

    # No gradient update
    with torch.no_grad():
        loss = 0
        for batch in iter(validation_loader):
            # Move data to the correct device
            target, source, label = batch
            target = target.to(device)
            source = source.to(device)
            label = label.to(device)

            prediction = model(target,source)
            batch_loss = criterion(prediction, label)
            loss += batch_loss.item()

            if device != torch.device('mps'):
                # make prediction of shape (batch_size, vocab_size) to shape (batch_size, 1, vocab_size) for perplexity calculation
                curr_batch_size = prediction.shape[0]
                ppl_pred = prediction.view(curr_batch_size, 1, -1)
                ppl_label = label.view(curr_batch_size, 1)

                perplexity.update(ppl_pred, ppl_label)
            multiclass_accuracy.update(prediction,label)
        # Average loss and accuracy over all batches
        loss /= len(validation_loader)
        if device != torch.device('mps'):
            perplexity = perplexity.compute()
        multiclass_accuracy = multiclass_accuracy.compute()
    # Log validation metrics
    if writer is not None:
        writer.add_scalar('Loss/val', loss, epoch)
        if device != torch.device('mps'):
            writer.add_scalar('Perplexity/val', perplexity, epoch)
        writer.add_scalar('Accuracy/val', multiclass_accuracy, epoch)
    
    return loss, perplexity if device != torch.device('mps') else 0, multiclass_accuracy


            