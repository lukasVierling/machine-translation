import sys
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import Perplexity, MulticlassAccuracy
from torchsummary import summary
from tqdm import tqdm
import numpy as np

from src.utils.pad_collate import pad_collate
from src.models import RNN_Encoder, RNN_Decoder
from data.dataset import Dataset_RNN
from src.preprocessing.dictionary import PADDING
from src.utils.get_optimizer import get_optimizer
from src.train import log

def rnn_train(config):
    print(torch.cuda.is_available())
    # Unpack configuration dictionary
    name = config['name']
    bpe_ops = config['bpe_ops']
    joined_bpe = config['joined_bpe']
    batch_size = config['batch_size']
    is_indexed = config['is_indexed']
    epochs = config['epochs']
    clip = config['clip']
    n_checkpoint = config['n_checkpoint']
    n_evaluate = config['n_evaluate']

    early_stopping = config['early_stopping']
    early_stopping_threshold = config['early_stopping_threshold']

    # load data paths
    train_source = f"../data/{bpe_ops}_{'joined_' if joined_bpe else ''}BPE_indexed/train/source.train.pickle"
    train_target = f"../data/{bpe_ops}_{'joined_' if joined_bpe else ''}BPE_indexed/train/target.train.pickle"
    dev_source = f"../data/{bpe_ops}_{'joined_' if joined_bpe else ''}BPE_indexed/dev/source.dev.pickle"
    dev_target = f"../data/{bpe_ops}_{'joined_' if joined_bpe else ''}BPE_indexed/dev/target.dev.pickle" 

    #load train and dev data
    print("Loading data...")
    train_dataset = Dataset_RNN(
        train_source,
        train_target,
        is_indexed=is_indexed,
        bpe_ops = bpe_ops,
        joined_bpe=joined_bpe
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    """
    DataLoader batchs are dictionaries with the keys
    'src': source sentences, padded to the length of the longest sentence in the batch with PAD_IDX
    'trg': target sentences, padded to the length of the longest sentence in the batch with PAD_IDX
    'src_lens': list of source sentence lengths
    'trg_lens': list of target sentence lengths
    """
    validation_dataset = Dataset_RNN(
        dev_source,
        dev_target,
        is_indexed=is_indexed,
        bpe_ops = bpe_ops,
        joined_bpe=joined_bpe
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    # get model
    print("Initializing model...")
    encoder, decoder = get_rnn_model(config, train_dataset)

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index = PADDING).to(config['device'])

    # Move the model to the given device
    device = config['device']
    print(f"Device: {device}")
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Model summary
    print(f"Encoder structure: \n{encoder}")
    print(f"Decoder structure: \n{decoder}")

    # create summary writer
    writer = SummaryWriter(log_dir=config['log_dir'] + f'/{name}/events')

    # optimizer
    print("Initializing optimizer...")
    encoder_optim = get_optimizer(encoder, config['optimizer'], config['learning_rate'],
                                  config['adam_beta1'], config['adam_beta2'],
                                  config['weight_decay'], config['momentum'],)
    decoder_optim = get_optimizer(decoder, config['optimizer'], config['learning_rate'],
                                    config['adam_beta1'], config['adam_beta2'],
                                    config['weight_decay'], config['momentum'])
    # learning rate halving scheduler
    if config['learning_rate_halving']:
        encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optim, mode='min', factor=0.5,
                                                                       patience=config['learning_rate_patience'],
                                                                       verbose=True,
                                                                       threshold=config['learning_rate_performance_epsilon'])
        decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optim, mode='min', factor=0.5,
                                                                       patience=config['learning_rate_patience'],
                                                                       verbose=True,
                                                                       threshold=config['learning_rate_performance_epsilon'])

    # Train the model
    starting_epoch = 0
    if config['resume_training']:
        starting_epoch = config['resume_epoch']
        print(f"Resuming training from epoch {starting_epoch}...")
    print(f"Training the model for {epochs} epochs...\n")
    for epoch in range(epochs):
        # tf ratio scheduling
        if config['teacher_forcing_schedule'] == 'exponential_decay':
            tf_ratio = config['teacher_forcing_ratio'] * (np.exp(-1 * epoch))
        elif config['teacher_forcing_schedule'] == 'linear_decay':
            tf_ratio = config['teacher_forcing_ratio'] * (1 - (epoch / epochs))
        elif config['teacher_forcing_schedule'] == 'constant':
            tf_ratio = config['teacher_forcing_ratio']
        else:
            raise ValueError(f"Invalid teacher_forcing_schedule: {config['teacher_forcing_schedule']}")

        loss, perplexity, multiclass_accuracy = train_epoch(encoder, decoder, criterion, encoder_optim, decoder_optim, train_loader, device, clip, tf_ratio, epoch)
        print(f"Epoch {epoch} loss (train): \t{loss}")
        print(f"Epoch {epoch} accuracy (train): \t{multiclass_accuracy}")
        print(f"Epoch {epoch} perplexity (train): \t{perplexity}")

        log(epoch, loss, perplexity, multiclass_accuracy, writer=writer, mode='train')

        # evaluate on validation set
        if (epoch + 1) % n_evaluate == 0:
            val_loss, val_perplexity, val_accuracy = rnn_evaluate(encoder, decoder, validation_loader, epoch, writer=writer, device=device)
            print(f"Epoch {epoch} loss (dev): \t{val_loss}")
            print(f"Epoch {epoch} accuracy (dev): \t{val_accuracy}")
            print(f"Epoch {epoch} perplexity (dev): \t{val_perplexity}")
            print("")


        # TODO add early stopping
        # Learning rate halving
        if config['learning_rate_halving']:
            encoder_scheduler.step(val_loss)
            decoder_scheduler.step(val_loss)

        # save model checkpoints if checkpoints_per_epoch is not set
        if config['checkpoints_per_epoch'] < 1 and epoch % n_checkpoint == 0:
            torch.save((encoder, decoder), config['log_dir'] + f'/{name}/checkpoints/model_{epoch}.pt')
            torch.save((encoder_optim, decoder_optim), config['log_dir'] + f'/{name}/optimizer/optimizer_{epoch}.pt')

    # save final model
    torch.save((encoder, decoder), config['log_dir'] + f'/{name}/checkpoints/model_final.pt')
    torch.save((encoder_optim, decoder_optim), config['log_dir'] + f'/{name}/optimizer/optimizer_final.pt')

    # close writer
    writer.close()

    print("Training finished.")

def train_epoch(encoder, decoder, criterion, encoder_optim, decoder_optim, train_loader, device, clip, tf_ratio, curr_epoch):
    """
    Function for encoder/decoder training a single epoch.

    Args:
        encoder (nn.Module): The encoder model.
        decoder (nn.Module): The decoder model.
        criterion (nn.Module): The loss function.
        encoder_optim (torch.optim): The optimizer for the encoder.
        decoder_optim (torch.optim): The optimizer for the decoder.
        train_loader (torch.data.utils.DataLoader): The data loader for the training data.
        device (torch.device): The device to use for training.
        clip (float): The gradient clipping value.
        curr_epoch (int): The current epoch.

    Returns:
        avg_loss_epoch (float): The average loss of the epoch.
        ppl (torch.Tensor | float): The perplexity of the epoch. (torch.Tensor if device is 'mps' else float('inf))
        acc (torch.Tensor): The accuracy of the epoch.
    """
    encoder.train()
    decoder.train()

    loss_epoch = 0
    multiclass_accuracy = MulticlassAccuracy(device=device)
    # perplexity currently not working with mps device
    if device != "mps":
        perplexity = Perplexity(device=device, ignore_index=PADDING)
    else:
        perplexity = None

    progress_bar = tqdm(train_loader, desc=f"Training Epoch {curr_epoch}...")
    for batch in progress_bar:
        # zero the gradients
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()

        # load data
        src = batch['src'].to(device)
        # src shape: (batch_size, max_src_length)
        trg = batch['trg'].to(device)
        # trg shape: (batch_size, max_trg_length)
        labels = batch['labels'].to(device)
        # labels shape: (batch_size, max_trg_length - 1)
        src_lens = batch['src_lens']
        trg_lens = batch['trg_lens']

        max_src_len = torch.max(src_lens).item()
        max_trg_len = torch.max(trg_lens).item()

        # calculate padding mask
        src_pad_mask = (src == PADDING).to(device)
        # src_pad_mask shape: (batch_size, max_src_length)

        # forward pass
        encoder_output, _ = encoder(src, src_lens)
        encoder_output = encoder_output.to(device)
        # encoder_output shape: (batch_size, max_src_len, encoder_hidden_size)
        decoder_outputs = decoder(encoder_output, max_src_len, max_trg_len, src_pad_mask, target_tensor=trg, teacher_forcing_ratio=tf_ratio, device=device).to(device)
        # decoder_output shape: (batch_size, target_vocab_size, (max_trg_length - 1))


        pred_indices = torch.argmax(decoder_outputs, dim=1)
        # pred_indices shape: (batch_size, max_trg_length - 1)
        #target_dict = train_dataset.get_target_dictionary()
        #pred_words = [[target_dict.getToken(index.item()) for index in sentence] for sentence in pred_indices]
        #for line in pred_words:
           #print(line)
        loss = criterion(decoder_outputs, labels)
        # loss shape: (batch_size, max_trg_length - 1)
        loss.backward()

        # clip gradients
        clip_grad_norm_(encoder.parameters(), clip)
        clip_grad_norm_(decoder.parameters(), clip)

        # update parameters
        encoder_optim.step()
        decoder_optim.step()

        # calculate metrics
        for i in range(max_trg_len - 1):
            # MCA cannot take multi-dimensional labels lol
            multiclass_accuracy.update(decoder_outputs[:,:,i], labels[:,i])
        #multiclass_accuracy.update(decoder_outputs, trg)
        if perplexity:
            # perplexity needs second dimensions to equal in contrast to loss smh
            ppl_pred = torch.permute(decoder_outputs, (0, 2, 1))
            perplexity.update(ppl_pred, labels)
        # update loss
        loss_epoch += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update()

    avg_loss_epoch = loss_epoch / len(train_loader)
    ppl = float('inf')
    if perplexity:
        ppl = perplexity.compute()
    acc = multiclass_accuracy.compute()
    
    return avg_loss_epoch, ppl, acc


def get_rnn_model(config, train_dataset):
    if config['model_path']:
        print(f"Loading model from {config['model_path']}")
        model = torch.load(config['model_path'])
        assert isinstance(model, tuple) and len(model) == 2, "Invalid model file"
        encoder, decoder = model
        return encoder, decoder
    encoder = RNN_Encoder(
        source_vocab_size=train_dataset.get_source_vocab_size(),
        rnn_hidden_size=config['encoder_rnn_hidden_size'],
        embedding_size=config['encoder_embedding_size'],
        num_rnn_layers=config['num_encoder_rnn_layers'],
        dropout_rate=config['encoder_dropout'],
        bidirectional=config['encoder_bidirectional'],
        config=config
    )
    decoder = RNN_Decoder(
        target_vocab_size=train_dataset.get_target_vocab_size(),
        rnn_hidden_size=config['decoder_rnn_hidden_size'],
        encoder_out_size=config['encoder_rnn_hidden_size'] * (2 if config['encoder_bidirectional'] else 1),
        num_rnn_layers=config['num_decoder_rnn_layers'],
        dropout_rate=config['decoder_dropout'],
        embedding_size=config['decoder_embedding_size'],
        attention_function=config['attention_function'],
        attention_lin_size=config['attention_lin_size'],
        teacher_forcing_ratio=config['teacher_forcing_ratio'],
        config=config
    )
    #decoder = RNN_Decoder(target_vocab_size, dec_hidden_size, encoder_out_size, num_layers, dropout, embedding_dim, attention_function, teacher_forcing_ratio)
    return encoder, decoder

def rnn_evaluate(encoder, decoder, validation_loader, epoch, writer=None, device=None):
    """
    Evaluate the encoder-decoder model on the validation dataset.

    Args:
        encoder (nn.Module): The encoder model.
        decoder (nn.Module): The decoder model.
        validation_loader: DataLoader for the validation dataset.
        epoch (int): Current epoch number.^
        writer (SummaryWriter or None): SummaryWriter object for logging metrics to TensorBoard, or None if not provided.
        device (torch.device or None): Device to use for evaluation, or None if not provided.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - loss (float): Average loss value on the validation dataset.
            - perplexity (float or None): Perplexity metric on the validation dataset, or None if device is 'mps'.
            - multiclass_accuracy (float): Multiclass accuracy on the validation dataset.
    """
    # Define loss function and metrics
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING).to(device)
    perplexity = None
    if device != "mps":
        perplexity = Perplexity(device=device, ignore_index=PADDING)
    multiclass_accuracy = MulticlassAccuracy(device=device)

    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()

    # Evaluate on the validation dataset
    with torch.no_grad():
        loss = 0
        for batch in validation_loader:
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            src_lens = batch['src_lens']#.to(device)
            trg_lens = batch['trg_lens']#.to(device)
            labels = batch['labels'].to(device)

            max_length_src = torch.max(src_lens)
            max_length_trg = torch.max(trg_lens)

            # calculate padding mask 
            # Fix: we have to change 1 to 0 and 0 to 1 because the padding mask is inverted
            src_pad_mask = (src == PADDING).to(device)

            encoder_output, _ = encoder(src, src_lens)
            encoder_output = encoder_output.to(device)
            decoder_outputs = decoder(encoder_output,
                                      max_length_src, 
                                      max_length_trg, 
                                      src_pad_mask, 
                                      prev_internal_state=None,
                                      target_tensor=None,
                                      device=device).to(device)
            
            batch_loss = criterion(decoder_outputs, labels)
            loss += batch_loss.item()
            
            #multiclass_accuracy.update(decoder_outputs, labels)
            if perplexity:
                # perplexity needs second dimensions to equal in contrast to loss smh
                ppl_pred = torch.permute(decoder_outputs, (0, 2, 1))
                perplexity.update(ppl_pred, labels)
            for i in range(max_length_trg - 1):
                # MCA cannot take multi-dimensional labels lol
                multiclass_accuracy.update(decoder_outputs[:,:,i], labels[:,i])
            #multiclass_accuracy.update(decoder_outputs, labels)

        avg_loss = loss / len(validation_loader)
        ppl = float('inf')
        if perplexity:
            ppl = perplexity.compute()
        acc = multiclass_accuracy.compute()

    # Log metrics to TensorBoard
    if writer:
        log(epoch, avg_loss, perplexity, multiclass_accuracy, bleu=0, writer=writer, mode='dev')
    
    return avg_loss, ppl, acc
if __name__ == "__main__":
    import yaml
    import os
    ROOT = "/Users/andreaspletschko/Documents/Uni/4. Semester/machine-translation"
    with open(ROOT + "/config/rnn_config.yml", 'r') as file:
        config = yaml.safe_load(file)
    
    print(sys.path)
    os.chdir("src")
    config['batch_size'] = 4
    config['name'] = "debugging"

    log_dir = config['log_dir']
    model_name = config['name']
    os.makedirs(log_dir + f'/{model_name}', exist_ok=True)
    os.makedirs(log_dir + f'/{model_name}/checkpoints', exist_ok=True)
    os.makedirs(log_dir + f'/{model_name}/optimizer', exist_ok=True)
    rnn_train(config)