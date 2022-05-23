import torch.nn as nn


class LSTMDecoder(nn.Module):
    """ Decodes hidden state output by encoder """

    def __init__(self, input_size, hidden_size, num_labels, dropout=0.0):
        """
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        """

        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, x_input, encoder_hidden_states):
        """
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        """

        lstm_out, hidden, cell = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        prediction = self.linear(lstm_out.squeeze(0))

        return prediction, hidden, cell
