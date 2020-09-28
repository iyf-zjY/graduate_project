import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse


class TPA_LSTM_modified(nn.Module):
    def __init__(self, args, original_columns, hs300=False, node_count=None):
        super(TPA_LSTM_modified, self).__init__()
        self.use_cuda = args.cuda
        self.hs300 = hs300
        self.node_count = node_count
        self.window_length = args.window;  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        self.original_columns = original_columns  # the number of columns or features
        self.hidR = args.hidRNN;
        self.hidden_state_features = args.hidden_state_features
        self.hidC = args.hidCNN;
        self.hidS = args.hidSkip;
        self.Ck = args.CNN_kernel;  # the kernel size of the CNN layers
        self.skip = args.skip;
        self.pt = (self.window_length - self.Ck) // self.skip
        self.hw = args.highway_window
        self.num_layers_lstm = args.num_layers_lstm
        self.hidden_state_features_uni_lstm = args.hidden_state_features_uni_lstm
        self.attention_size_uni_lstm = args.attention_size_uni_lstm
        self.num_layers_uni_lstm = args.num_layers_uni_lstm
        self.lstm = nn.LSTM(input_size=self.original_columns, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            bidirectional=False);
        self.uni_lstm = nn.LSTM(input_size=1, hidden_size=args.hidden_state_features_uni_lstm,
                                num_layers=args.num_layers_uni_lstm,
                                bidirectional=False);
        self.compute_convolution = nn.Conv2d(1, self.hidC, kernel_size=(
            self.Ck, self.hidden_state_features))  # hidC are the num of filters, default value of Ck is one

        self.fc = nn.Linear(in_features=self.original_columns, out_features=2)
        self.attention_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidC, self.hidden_state_features,
                       requires_grad=True)).cuda()  # , device='cuda'
        self.context_vector_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidC,
                       requires_grad=True)).cuda()  # , device='cuda'
        self.final_state_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features,
                       requires_grad=True)).cuda()  # , device='cuda'
        self.final_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.original_columns, self.hidden_state_features,
                       requires_grad=True)).cuda()  # , device='cuda'

        self.attention_matrix_uni_lstm = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm,
                       self.original_columns, requires_grad=True)).cuda()
        self.context_vector_matrix_uni_lstm = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True)).cuda()
        self.final_hidden_uni_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features_uni_lstm, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True)).cuda()
        self.final_uni_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features_uni_lstm,
                       self.original_columns,
                       requires_grad=True)).cuda()

        self.bridge_matrix = nn.Parameter(
            torch.ones(args.batch_size, self.hidden_state_features, self.hidden_state_features,
                       requires_grad=True)).cuda()

        self.attn_weight = None if not self.hs300 else nn.Parameter(torch.Tensor(1, self.node_count)).cuda()

        torch.nn.init.xavier_uniform(self.attention_matrix)
        torch.nn.init.xavier_uniform(self.context_vector_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.final_matrix)
        torch.nn.init.xavier_uniform(self.attention_matrix_uni_lstm)
        torch.nn.init.xavier_uniform(self.context_vector_matrix_uni_lstm)
        torch.nn.init.xavier_uniform(self.final_hidden_uni_matrix)
        torch.nn.init.xavier_uniform(self.final_state_matrix)
        torch.nn.init.xavier_uniform(self.bridge_matrix)

        self.conv1 = nn.Conv2d(1, self.hidC,
                               kernel_size=(self.Ck, self.original_columns));  # kernel size is size for the filters
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.original_columns);
        else:
            self.linear1 = nn.Linear(self.hidR, self.original_columns);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x, adj):
        batch_size = x.size(0);
        x = x.cuda()

        """
           Step 1. First step is to feed this information to LSTM and find out the hidden states

            General info about LSTM:

            Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for `t = seq_len`

        """
        ##Incase in future bidirectional lstms are to be used, size of hn would needed to be modified a little (as output is of size (num_layers * num_directions, batch, hidden_size))
        input_to_lstm = x.permute(1, 0,
                                  2).contiguous()  # input to lstm is of shape (seq_len, batch, features) (x is of shape (batch_size, seq_length, features))
        lstm_hidden_states, (h_all, c_all) = self.lstm(input_to_lstm)
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))

        """
            Step 2. Apply convolution on these hidden states. As in the paper TPA-LSTM, these filters are applied on the rows of the hidden state
        """
        output_realigned = lstm_hidden_states.permute(1, 0, 2).contiguous()
        hn = hn.permute(1, 0, 2).contiguous()
        # cn = cn.permute(1, 0, 2).contiguous()
        input_to_convolution_layer = output_realigned.view(-1, 1, self.window_length, self.hidden_state_features);
        convolution_output = F.relu(self.compute_convolution(input_to_convolution_layer));
        convolution_output = self.dropout(convolution_output);

        """
            Step 3. Apply attention on this convolution_output
        """
        convolution_output = convolution_output.squeeze(3)

        """
                In the next 10 lines, padding is done to make all the batch sizes as the same so that they do not pose any problem while matrix multiplication
                padding is necessary to make all batches of equal size
        """
        final_hn = torch.zeros(self.attention_matrix.size(0), 1, self.hidden_state_features)
        input = torch.zeros(self.attention_matrix.size(0), x.size(1), x.size(2))
        final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.hidC, self.window_length)
        diff = 0
        if (hn.size(0) < self.attention_matrix.size(0)):
            final_hn[:hn.size(0), :, :] = hn
            final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
            input[:x.size(0), :, :] = x
            diff = self.attention_matrix.size(0) - hn.size(0)
        else:
            final_hn = hn
            final_convolution_output = convolution_output
            input = x.cuda()

        """
           final_hn, final_convolution_output are the matrices to be used from here on
        """
        convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous()
        final_hn_realigned = final_hn.permute(0, 2, 1).contiguous()
        convolution_output_for_scoring = convolution_output_for_scoring.cuda()
        final_hn_realigned = final_hn_realigned.cuda()
        mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix).cuda()
        scoring_function = torch.bmm(mat1, final_hn_realigned).cuda()
        alpha = torch.nn.functional.sigmoid(scoring_function)
        context_vector = alpha * convolution_output_for_scoring
        context_vector = torch.sum(context_vector, dim=1).cuda()

        """
           Step 4. Compute the output based upon final_hn_realigned, context_vector
        """
        context_vector = context_vector.view(-1, self.hidC, 1)
        h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix,
                                                                                            context_vector)

        """
            Up until now TPA-LSTM has been implemented in pytorch

            Modification
            Background: TPA-LSTM guys use all features together and stack them up during the RNN stage. This treats one time step as one entity, killing the individual
            properties of each variable. At the very end, the variable we are predicting for must depend on itself the most.

            Proposition: Use lstm on each variable independently (assuming them to be independent). Using the hidden state of each time step, along with the hidden state
            of the CNN, now apply the same attention model. This method will preserve the identiy of the individual series by not considering all of them as one state.   
        """
        individual_all_hidden_states = None
        individual_last_hidden_state = None
        for feature_num in range(0, self.original_columns):
            individual_feature = input[:, :, feature_num].view(input.size(0), input.size(1), -1).permute(1, 0,
                                                                                                         2).contiguous().cuda()
            uni_output, (uni_hn, uni_cn) = self.uni_lstm(
                individual_feature)  # Output of hn is of the size  (num_layers * num_directions, batch, hidden_size)  |  num_layers = 2 in bidirectional lstm
            if (feature_num == 0):
                individual_all_hidden_states = uni_output.permute(1, 0, 2).contiguous()
                individual_last_hidden_state = uni_hn[-1].view(1, uni_hn.size(1), uni_hn.size(2)).permute(1, 0,
                                                                                                          2).contiguous()
            else:
                individual_all_hidden_states = torch.cat(
                    (individual_all_hidden_states, uni_output.permute(1, 0, 2).contiguous()), 1)
                individual_last_hidden_state = torch.cat((individual_last_hidden_state,
                                                          uni_hn[-1].view(1, uni_hn.size(1), uni_hn.size(2)).permute(1,
                                                                                                                     0,
                                                                                                                     2).contiguous()),
                                                         1)

        ## *****************DIMENSIONS OF individual_all_hidden_states are (batch_size, time series length/window size, hidden_state_features, total univariate series)*****************##
        ## *****************DIMENSIONS OF individual_last_hidden_state are (batch_size, 1, hidden_state_features, total univariate series)*****************##

        individual_all_hidden_states = individual_all_hidden_states.view(input.size(0), input.size(1),
                                                                         self.hidden_state_features_uni_lstm, -1).cuda()
        individual_last_hidden_state = individual_last_hidden_state.view(input.size(0), 1,
                                                                         self.hidden_state_features_uni_lstm, -1).cuda()
        """
        Calculate the attention score for all of these
        """
        univariate_attended = []
        h_output = None
        for feature_num in range(0, self.original_columns):
            attention_matrix_uni = self.attention_matrix_uni_lstm[:, :, :, feature_num]
            context_vector_matrix_uni = self.context_vector_matrix_uni_lstm[:, :, :, feature_num]
            hidden_matrix = self.final_hidden_uni_matrix[:, :, :, feature_num]
            final_matrix = self.final_uni_matrix[:, :, :, feature_num]
            all_hidden_states_single_variable = individual_all_hidden_states[:, :, :, feature_num].cuda()
            final_hidden_state = individual_last_hidden_state[:, :, :, feature_num].permute(0, 2, 1).contiguous().cuda()

            mat1 = torch.bmm(all_hidden_states_single_variable, attention_matrix_uni).cuda()
            mat2 = torch.bmm(mat1, final_hidden_state).cuda()
            attention_score = torch.sigmoid(mat2).cuda()

            context_vector_individual = attention_score * all_hidden_states_single_variable
            context_vector_individual = torch.sum(context_vector_individual, dim=1).cuda()
            context_vector_individual = context_vector_individual.view(context_vector_individual.size(0),
                                                                       context_vector_individual.size(1), 1).cuda()

            attended_states = torch.bmm(context_vector_matrix_uni, context_vector_individual).cuda()
            h_intermediate1 = attended_states + torch.bmm(hidden_matrix, final_hidden_state).cuda()

            if (feature_num == 0):
                h_output = torch.bmm(final_matrix, h_intermediate1).cuda()
            else:
                h_output += torch.bmm(final_matrix, h_intermediate1).cuda()

        h_intermediate2 = torch.bmm(self.bridge_matrix, h_output).cuda()

        """
           Combining the two
        """
        h_intermediate = h_intermediate + h_intermediate2
        result = torch.bmm(self.final_matrix, h_intermediate.cuda())
        result = result.permute(0, 2, 1).contiguous()
        result = result.squeeze()

        """
           Remove from result the extra result points which were added as a result of padding
        """
        final_result = result[:result.size(0) - diff]

        """
        Adding highway network to it
        """

        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1, self.original_columns);
            res = final_result + z;
        # return nn.functional.softmax(self.fc(res), dim=1)
        return torch.matmul(nn.functional.softmax(self.attn_weight, dim=1), nn.functional.softmax(self.fc(res), dim=1)) \
            if self.hs300 else nn.functional.softmax(self.fc(res), dim=1)
        # return torch.sigmoid(res)


def arg_parse_TPA():
    parser_1 = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser_1.add_argument('--data', type=str, default="data/data_stock.txt",
                          help='location of the data file')
    # , required=True
    parser_1.add_argument('--model', type=str, default='TPA_LSTM_Modified',
                          help='')
    parser_1.add_argument('--hidden_state_features', type=int, default=30,
                          help='number of features in LSTMs hidden states')
    parser_1.add_argument('--num_layers_lstm', type=int, default=1,
                          help='num of lstm layers')
    parser_1.add_argument('--hidden_state_features_uni_lstm', type=int, default=1,
                          help='number of features in LSTMs hidden states for univariate time series')
    parser_1.add_argument('--num_layers_uni_lstm', type=int, default=1,
                          help='num of lstm layers for univariate time series')
    parser_1.add_argument('--attention_size_uni_lstm', type=int, default=10,
                          help='attention size for univariate lstm')
    parser_1.add_argument('--hidCNN', type=int, default=10,
                          help='number of CNN hidden units')
    parser_1.add_argument('--hidRNN', type=int, default=100,
                          help='number of RNN hidden units')
    parser_1.add_argument('--window', type=int, default=30,
                          help='window size')
    parser_1.add_argument('--CNN_kernel', type=int, default=1,
                          help='the kernel size of the CNN layers')
    parser_1.add_argument('--highway_window', type=int, default=24,
                          help='The window size of the highway component')
    parser_1.add_argument('--clip', type=float, default=10.,
                          help='gradient clipping')
    parser_1.add_argument('--epochs', type=int, default=3000,
                          help='upper epoch limit')  # 30
    parser_1.add_argument('--batch_size', type=int, default=295, metavar='N',
                          help='batch size')
    parser_1.add_argument('--dropout', type=float, default=0.1,
                          help='dropout applied to layers (0 = no dropout)')
    parser_1.add_argument('--seed', type=int, default=54321,
                          help='random seed')
    parser_1.add_argument('--gpu', type=int, default=None)
    parser_1.add_argument('--log_interval', type=int, default=2000, metavar='N',
                          help='report interval')
    parser_1.add_argument('--save', type=str, default='model/model.pt',
                          help='path to save the final model')
    parser_1.add_argument('--cuda', type=str, default=True)
    parser_1.add_argument('--optim', type=str, default='adam')
    parser_1.add_argument('--lr', type=float, default=0.005)
    parser_1.add_argument('--momentum', type=float, default=0.2)
    parser_1.add_argument('--horizon', type=int, default=1)  # prediction window size
    parser_1.add_argument('--skip', type=float, default=24)
    parser_1.add_argument('--hidSkip', type=int, default=5)
    parser_1.add_argument('--L1Loss', type=bool, default=False)
    parser_1.add_argument('--normalize', type=int, default=0)
    parser_1.add_argument('--output_fun', type=str, default='sigmoid')
    args_1 = parser_1.parse_args()
    return args_1

