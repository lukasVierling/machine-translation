import sys
sys.path.append("..")
sys.path.append("../..")
#sys.path.append("/Users/andreaspletschko/Documents/Uni/4. Semester/machine-translation")

import torch
import numpy as np
from torch import nn
from torch.nn.functional import softmax, relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from random import random


from src.preprocessing.dictionary import START, END, PADDING
PADDING = 0

MASK_VALUE = -1e10

from torchsummary import summary

class RNN_Encoder(nn.Module):
    def __init__(self, source_vocab_size, rnn_hidden_size, embedding_size,
                 num_rnn_layers, dropout_rate, bidirectional, 
                 rnn_type='lstm', config=None):
        """
        Encoder class for the RNN model.

        Args:
            source_vocab_size (int): Size of the source vocabulary.
            hidden_size (int): Size of the LSTM hidden layer(s).
            embedding_dim (int): Dimension of the embedding layer (i.e. word embeddings).
            num_lstm_layers (int): Number of LSTM layers.
            lstm_dropout_rate (float): Dropout rate for the LSTM layers.
            bidirectional (bool): Whether to use a bidirectional LSTM.
        """
        super(RNN_Encoder, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(source_vocab_size, embedding_size, padding_idx=PADDING)
        # Embedding normalization
        #self.embedding_norm = nn.LayerNorm(embedding_size)
        # LSTM layer
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_size, 
                            hidden_size=rnn_hidden_size, 
                            num_layers=num_rnn_layers, 
                            batch_first=True, 
                            bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_size, 
                            hidden_size=rnn_hidden_size, 
                            num_layers=num_rnn_layers, 
                            batch_first=True, 
                            bidirectional=bidirectional)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=embedding_size, 
                            hidden_size=rnn_hidden_size, 
                            num_layers=num_rnn_layers, 
                            batch_first=True,  
                            bidirectional=bidirectional)
        else:
            raise ValueError("RNN type not supported")
        
        # Layer Norm
        #self.rnn_norm = nn.LayerNorm(rnn_hidden_size*2 if bidirectional else rnn_hidden_size)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        self.config = config
    
    def forward(self, padded_src, src_lens):
        """
        Forward pass through the encoder.

        Args:
            padded_src (torch.Tensor): Padded source sequences. 
                Expected shape: (batch_size, max_sequence_length) 
            src_lens (torch.Tensor): Lengths of the source sequences. 
                Expected shape: (batch_size,)

        Returns:
            padded_output (torch.Tensor): Padded output of the encoder. 
                Expected shape: (batch_size, max_sequence_length, hidden_size * n_directions)
            state (torch.Tensor, torch.Tensor): Tuple of the last hidden state and cell state of the LSTM encoder. 
                Expected shapes: (num_lstm_layers * n_directions, batch_size, hidden_size)
        """
        # Embedding layer
        padded_embedded = self.embedding(padded_src)
        # Layer Norm
        #padded_embedded = self.embedding_norm(padded_embedded)

        dropped_embedded = self.dropout(padded_embedded)
        # padded_embedded dim: batch_size x max_sequence_length x embedding_dim
        packed_embedded = pack_padded_sequence(input=dropped_embedded, 
                                               lengths=src_lens, 
                                               batch_first=True, 
                                               enforce_sorted=False)
        # LSTM layer
        packed_output, state = self.rnn(packed_embedded)
        padded_output, _ = pad_packed_sequence(sequence=packed_output, 
                                               batch_first=True, 
                                               padding_value=PADDING)
        # Layer Norm
        #padded_output = self.rnn_norm(padded_output)
        
        return padded_output, state


class RNN_Decoder(nn.Module):
    def __init__(self, target_vocab_size, rnn_hidden_size,  
                 encoder_out_size, num_rnn_layers, dropout_rate, 
                 embedding_size, attention_function, attention_lin_size, 
                 rnn_type='lstm', teacher_forcing_ratio = 0.0, config=None):
        """
        Decoder class for the RNN model.

        Args:
            target_vocab_size (int): Size of the target vocabulary.
            decoder_hidden_size (int): Size of the LSTM hidden layer(s).
            encoder_out_size (int): Size of the encoder output (i.e. encoder_hidden_size * n_directions).
            num_lstm_layers (int): Number of LSTM layers.
            lstm_dropout_rate (float): Dropout rate for the LSTM layers.
            embedding_dim (int): Dimension of the embedding layer (i.e. word embeddings).
            attention_function (str): Attention function to use. Possible values: "dot_product", "additive".
            attention_lin_size (int): Size of the linear layer for the "additive" attention function. Ignored for the "dot_product" attention function.
            teacher_forcing_ratio (float): Ratio of teacher forcing to use during training.
        """
        super(RNN_Decoder, self).__init__()
        
        assert teacher_forcing_ratio >= 0.0 and teacher_forcing_ratio <= 1.0, "Teacher forcing ratio must be between 0.0 and 1.0"
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.decoder_rnn_hidden_size = rnn_hidden_size
        self.target_vocab_size = target_vocab_size
        self.num_rnn_layers = num_rnn_layers

        # Embedding layer
        self.embedding = nn.Embedding(target_vocab_size, embedding_size, padding_idx=PADDING)
        # Layer Norm
        #self.embedding_norm = nn.LayerNorm(embedding_size)

        # Attention layer
        self.attention = AttentionLayer(attention=attention_function, 
                                        encoder_out_size=encoder_out_size, 
                                        decoder_hidden_size=rnn_hidden_size, 
                                        lin_size=attention_lin_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        
        # RNN layer
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_size + encoder_out_size, 
                            hidden_size=rnn_hidden_size, 
                            num_layers=num_rnn_layers, 
                            batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_size + encoder_out_size, 
                            hidden_size=rnn_hidden_size, 
                            num_layers=num_rnn_layers, 
                            batch_first=True)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=embedding_size + encoder_out_size, 
                            hidden_size=rnn_hidden_size, 
                            num_layers=num_rnn_layers, 
                            batch_first=True)
        else:
            raise ValueError("RNN type not supported")
        # Output layer
        self.rnn_type = rnn_type
        # RNN norm
        #self.rnn_norm = nn.LayerNorm(rnn_hidden_size)
    
        self.lin = nn.Linear(in_features=rnn_hidden_size, 
                             out_features=target_vocab_size)
        
        self.config = config

    def forward(self, encoder_output, max_length_src, 
                max_length_trg, mask,
                prev_internal_state=None, target_tensor=None, 
                teacher_forcing_ratio=0.0, device='cpu'):
        """
        Forward pass through the decoder.

        Args:
            encoder_output (torch.Tensor): Output of the encoder, i.e. all last layer hidden states (concatenated if bidirectional) of the encoder. 
                Expected shape: (batch_size, max_src_length, encoder_hidden_size * n_directions)
            max_length_src (int): Maximum length of the source sequences.
            max_length_trg (int): Maximum length of the target sequences.
            mask (torch.Tensor): Mask for the attention layer for ignoring padded positions.
                Expected shape: (batch_size, max_sequence_length)
            prev_internal_state (torch.Tensor): Previous internal state of the decoder RNN 
                (i.e. hidden state for GRU and vanilla RNN or tuple of cell state and hidden state for LSTM).
                Expected shapes: (num_lstm_layers, batch_size, decoder_hidden_size) or None
            target_tensor (torch.Tensor): Target sequences. Used for teacher forcing.
                Expected shape: (batch_size, max_trg_length)
        
        Returns:
            decoder_outputs (torch.Tensor): Outputs of the decoder, i.e. the logits for each word in the target vocabulary at each time step.
                Expected shape: (batch_size, target_vocab_size, (max_trg_length - 1))
        """
        #Get batch size
        batch_size = encoder_output.size(0)

        # Initialize hidden state and cell state
        if prev_internal_state is None:
            if self.rnn_type == 'lstm':
                prev_hidden = torch.zeros(self.num_rnn_layers, batch_size, self.decoder_rnn_hidden_size).to(device)
                prev_cell_state = torch.zeros(self.num_rnn_layers, batch_size, self.decoder_rnn_hidden_size).to(device)
                prev_internal_state = (prev_hidden, prev_cell_state)
            else:
                prev_hidden = torch.zeros(self.num_rnn_layers, batch_size, self.decoder_rnn_hidden_size).to(device)
                prev_internal_state = prev_hidden
        
        # assert that we got the right format for prev_internal_state
        lstm_condition = (self.rnn_type == 'lstm' and isinstance(prev_internal_state, tuple) and len(prev_internal_state) == 2)
        gru_condition = (self.rnn_type != 'lstm' and isinstance(prev_internal_state, torch.Tensor))
        assert lstm_condition or gru_condition, "prev_internal_state has wrong format"

        # initialize decoder predictions with START toke 
        decoder_prediction = torch.zeros(batch_size, 1).fill_(START).to(dtype=torch.int32)
        # decoder_predictions dim: batch_size x 1

        # decoder_outputs will contain the output of the decoder for each timestep
        decoder_outputs = torch.empty(batch_size, self.target_vocab_size, (max_length_trg - 1))
        # decoder_outputs dim: batch_size x target_vocab_size x (max_length - 1) 

        # max_length_trg - 1 because decoder does not predict SOS token
        for i in range(max_length_trg-1):
            # choose next input word for decoder (probabilistic teacher forcing)
            if target_tensor is not None and random() < teacher_forcing_ratio:
                input = target_tensor[:,i].unsqueeze(1)
            else:
                input = decoder_prediction
            # input dim: batch_size x 1
            input = input.to(device)

            # Forward step through the decoder
            decoder_output, prev_internal_state = self.forward_step(input,
                                                                    encoder_output, 
                                                                    prev_internal_state,
                                                                    max_length_src, 
                                                                    mask,device)
            # decoder_output dim: batch_size x 1 x target_vocab_size

            # Create prob. distribution to get the next predicted word
            # dim=2 because we have batch_size x seq_len (= 1) x target_vocab_size
            probabilities = softmax(decoder_output,dim=2)
            # probabilities dim: batch_size x 1 x target_vocab_size
            
            # Get the indexes of the word with the highest probability (for each batch)
            _, index = torch.max(probabilities, dim=2)
            # index dim: batch_size x 1

            decoder_prediction = index
            # decoder_predictions dim: batch_size x 1

            # Add prediction to the predictions tensor
            # remember to squeeze the decoder_output to get rid of the seq_len = 1 dimension
            decoder_outputs[:, :, i] = decoder_output.squeeze(1)
            
        return decoder_outputs
            
    def forward_step(self, input, encoder_output, prev_internal_state, max_length_src, mask,device='cpu'):
        """
        Single forward step through the decoder.

        Args:
            input (torch.Tensor): Input to the decoder.
                Expected shape: (batch_size, 1)
            encoder_output (torch.Tensor): Output of the encoder, i.e. all hidden states of the encoder.
                Expected shape: (batch_size, max_sequence_length, encoder_hidden_size * n_directions)
            prev_internal_state (torch.Tensor): Previous internal state of the decoder LSTM (i.e. tuple of cell state and hidden state).
                Expected shapes: (num_lstm_layers, batch_size, decoder_hidden_size)
            max_length_src (int): Maximum length of the source sequences.
            mask (torch.Tensor): Mask for the attention layer for ignoring padded positions.
                Expected shape: (batch_size, max_sequence_length)

        Returns:
            output (torch.Tensor): Output of the decoder.
                Expected shape: (batch_size, 1, target_vocab_size)
            (hidden, cell_state) (torch.Tensor, torch.Tensor)/hidden (torch.Tensor): Tuple of the last hidden state and cell state of the decoder LSTM
                or only hidden state if something other than LSTM has been used.
                Expected shape: (num_lstm_layers, batch_size, decoder_hidden_size)
        """
        # unpack internal state
        if self.rnn_type == 'lstm':
            prev_hidden, _ = prev_internal_state
        else:
            prev_hidden = prev_internal_state

        # Embedding layer
        embedded_input = self.embedding(input)
        #embedded_input = self.embedding_norm(embedded_input)
        # embedded input dim: batch_size x 1 x embedding_dim
    
        # Attention layer
        # prev_hidden has to be altered to match the dimensions of the attention layer
        # only select the hidden state of the last layer
        last_hidden = torch.permute(prev_hidden,(1,0,2))[:,-1,:].squeeze(1)
        # last_hidden dim: batch_size x  decoder_hidden_size
        context = self.attention(encoder_output, last_hidden, max_length_src, mask, device)
        # Context dim: batch_size x 1 x (encoder_output.size * n_directions)
        # Concatenation Layer 
        concat_input = torch.cat((embedded_input, context), dim=2)
        # concat input dim: batch_size x 1 x (embedding_dim + encoder_out_size)

        # LSTM Layer
        rnn_output, internal_state = self.rnn(concat_input, prev_internal_state)
        # rnn output dim: batch_size x 1 x decoder_hidden_size
        # hidden, cell_state dim: num_lstm_layers  x batch_size x decoder_hidden_size
        # Dropout layer
        #rnn_output = self.rnn_norm(rnn_output)
        rnn_output_dropped = self.dropout(rnn_output)
        # Output layer
        output = self.lin(rnn_output_dropped)
        # Output dim: batch_size x 1 x target_vocab_size
        # Return
        return output, internal_state


class AttentionLayer(nn.Module):
    def __init__(self, attention, encoder_out_size, decoder_hidden_size, lin_size):
        """
        Attention layer for the RNN model. Both additive (Bahdanau) and dot product attention are implemented.

        Args:
            attention (str): Attention function to use. Possible values: "dot_product", "additive".
            encoder_out_size (int): Size of the encoder output (i.e. encoder_hidden_size * n_directions * n_layer).
            decoder_hidden_size (int): Size of the LSTM hidden layer(s).
            lin_dim (int): Size of the linear layer for the "additive" attention function. Ignored for the "dot_product" attention function.
        """
        super(AttentionLayer, self).__init__()
        self.ignore_attention = False
        if attention == "dot_product":
            self.linQ = nn.Linear(decoder_hidden_size, encoder_out_size)
            self.linK = nn.Linear(encoder_out_size, encoder_out_size)
            self.calculate_energy = self.dot_product

        elif attention == "additive":
            print(f"encoder_out_size + decoder_hidden_size: {encoder_out_size+decoder_hidden_size}")
            self.lin1 = nn.Linear(encoder_out_size + decoder_hidden_size, lin_size)
            self.lin2 = nn.Linear(lin_size, 1)
            self.calculate_energy = self.additive
        elif attention == "none":
            self.ignore_attention = True
        else:
            raise ValueError("Attention function not implemented")
        
    def forward(self, encoder_output, decoder_hidden_state, max_src_length, mask, device):
        """
        Forward pass through the attention layer.

        Args:
            encoder_output (torch.Tensor): Output of the encoder, i.e. all last layer hidden states of the encoder.
                Expected shape: (batch_size, max_src_length, encoder_hidden_size * n_directions)
            decoder_hidden_state (torch.Tensor): Last layer hidden state of the decoder.
                Expected shape: (batch_size, decoder_hidden_size)
            max_src_length (int): Maximum length of the source sequences.
            mask (torch.Tensor): Mask for the attention layer for ignoring padded positions.
                Expected shape: (batch_size, max_sequence_length)

        Returns:
            context (torch.Tensor): Context vector.
                Expected shape: (batch_size, 1, encoder_out_size)
        """
        if self.ignore_attention:
            return encoder_output[:, -1, :].unsqueeze(1)
            # return dim: batch_size x 1 x encoder_out_size
    
        energy = torch.zeros((encoder_output.size(0), max_src_length)).to(device)
        # energy dim: batch_size x max_src_length

        for i in range(max_src_length):
            # calculate energy for i-th encoder output
            energy_vec = self.calculate_energy(encoder_output[:,i,:], decoder_hidden_state).to(device)
            # energy_vec dim: batch_size x 1
            # mind possible non-deterministic index error

            energy[:,i] = energy_vec.squeeze(1)

        # mask the attention values before softmaxing to ignore padding
        energy = energy.masked_fill(mask, value=MASK_VALUE)
        # mind possible non-deterministic runtime error

        alphas = softmax(energy, dim=1).unsqueeze(1).to(device)
        # alphas dim: batch_size x 1 x max_src_length

        # calculate context vector as weighted sum of encoder outputs
        context = torch.bmm(alphas, encoder_output).to(device)
        # context dim: batch_size x 1 x encoder_out_size

        return context
    
    def additive(self, encoder_hidden_state, decoder_hidden_state):
        """
        Calculation of the energy for the additive attention function.

        Args:
            encoder_hidden_state (torch.Tensor): Last layer hidden state at a certain sentence position in the encoder (f_j's for every batch).
                Expected shape: (batch_size, encoder_out_size), where encoder_out_size = encoder_hidden_size * n_directions
            decoder_hidden_state (torch.Tensor): Last layer hidden state of the decoder (s_i for every batch).
                Expected shape: (batch_size, decoder_hidden_size)
        """
        # Concat Layer
        #TODO eigener lin layer für states danach cat?
        input = torch.cat((encoder_hidden_state, decoder_hidden_state), dim=1)
        # input dim: batch_size x (encoder_out_size + decoder_hidden_size)
        # 1st Lin Layer
        lin_input = self.lin1(input)
        # lin_input dim: batch_size x lin_dim

        # activation function
        relued_input = relu(lin_input)

        # 2nd Lin Layer
        output = self.lin2(relued_input)
        # output dim: batch_size x 1

        return output

    def dot_product(self, encoder_hidden_state, decoder_hidden_state):
        # Dimensionen müssen stimmen lol
        queries = self.linQ(decoder_hidden_state)
        keys = self.linK(encoder_hidden_state)
        batch_size = queries.size(0)
        hidden_dim = queries.size(1)
        #remove last unnecessary dim
        # Add scaling factor
        unscaled = torch.bmm(queries.view(batch_size,1,hidden_dim), keys.view(batch_size,hidden_dim,1)).squeeze(2)
        # should be sqrt(d_model) but maybe this also works lol
        scaled = unscaled/np.sqrt(hidden_dim)
        return scaled

if __name__ == "__main__":
    # Define input tensors
    source_vocab_size = 7000
    encoder_hidden_size = 100
    encoder_embedding_dim = 64
    num_encoder_layers = 2
    lstm_dropout_rate = 0.1
    bidirectional = True

    encoder = RNN_Encoder(source_vocab_size, encoder_hidden_size, encoder_embedding_dim,
                          num_encoder_layers, lstm_dropout_rate, bidirectional)

    # Generate random input data
    batch_size = 8
    max_sequence_length = 10
    source_lens = torch.randint(1, max_sequence_length+1, (batch_size,))
    print(source_lens.shape)
    sequences = [torch.randint(0, source_vocab_size, (source_lens[i].item(),)) for i in range(batch_size)]
    padded_src = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PADDING)
    print(f"{padded_src.shape}, Expected: ({batch_size}, {max_sequence_length})") # Expected: (batch_size, max_sequence_length)
    max_soure_length = torch.max(source_lens).item()
    # Forward pass through the encoder
    padded_output, (h, c) = encoder(padded_src, source_lens)

    # Check output shapes
    print(f"{padded_output.shape}, Expected: ({batch_size}, {max_sequence_length}, {encoder_hidden_size * (2 if bidirectional else 1)})")  # Expected: (batch_size, max_sequence_length, hidden_size * n_directions)
    print(f"{h.shape}, Expected: ({num_encoder_layers * (2 if bidirectional else 1)}, {batch_size}, {encoder_hidden_size})")  # Expected: (num_lstm_layers * n_directions, batch_size, hidden_size)
    print(f"{c.shape}, Expected: ({num_encoder_layers * (2 if bidirectional else 1)}, {batch_size}, {encoder_hidden_size})")  # Expected: (num_lstm_layers * n_directions, batch_size, hidden_size)

    ##### Decoder Test #####
    print("\n\nDecoder Test\n")
    # Define input tensors
    target_vocab_size = 7000
    decoder_hidden_size = 100
    encoder_out_size = (2 if bidirectional else 1) * encoder_hidden_size 
    num_decoder_layers = 2
    lstm_dropout_rate = 0.1
    decoder_embedding_dim = 64
    attention_function = "additive"
    attention_lin_size = 50
    teacher_forcing_ratio = 1

    decoder = RNN_Decoder(target_vocab_size, decoder_hidden_size, encoder_out_size,
                          num_decoder_layers, lstm_dropout_rate, decoder_embedding_dim,
                          attention_function, attention_lin_size, teacher_forcing_ratio)

    #summary(encoder)
    #summary(decoder)

    # Generate random input data
    max_sequence_length = 5
    encoder_output = padded_output
    mask = (padded_src == PADDING)
    #prev_hidden = torch.randn(num_layers, batch_size, decoder_hidden_size)
    prev_hidden = None
    target_lens = torch.randint(1, max_sequence_length+1, (batch_size,))
    print(f"Target lens: {target_lens}")
    target_seqs = [torch.randint(0, target_vocab_size, (target_lens[i].item(),)) for i in range(batch_size)]
    padded_target = torch.nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=PADDING)
    print(f"Target shape: {padded_target.shape}")
    max_target_length = torch.max(target_lens).item()

    # Forward pass through the decoder
    decoder_outputs = decoder(encoder_output, max_soure_length, max_target_length,
                              mask, prev_hidden, padded_target)

    if max_target_length != max_sequence_length:
        print(f"llolol")
    # Check output shape
    print(decoder_outputs.shape)  # Expected: (batch_size, max_sequence_length, target_vocab_size)
