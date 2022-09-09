def get_model():
    import torch
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    import math
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=0.1)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return x

    class SelfAttentionPooling(nn.Module):
        """
        Implementation of SelfAttentionPooling 
        Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
        https://arxiv.org/pdf/2008.01077v1.pdf
        """
        def __init__(self, input_dim):
            super(SelfAttentionPooling, self).__init__()
            self.W = nn.Linear(input_dim, 1)
            
        def forward(self, batch_rep):
            """
            input:
                batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
            
            attention_weight:
                att_w : size (N, T, 1)
            
            return:
                utter_rep: size (N, H)
            """
            softmax = nn.functional.softmax
            att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
            utter_rep = torch.sum(batch_rep * att_w, dim=1)

            return utter_rep

    class TransformerModel(nn.Module):

        def __init__(self, d_model, nhead, dim_feedforward, nlayers, n_conv_layers=2, n_class=2, dropout=0.5, dropout_other=0.1):
            super(TransformerModel, self).__init__()
            self.model_type = 'Transformer'
            self.n_class = n_class
            self.n_conv_layers = n_conv_layers
            self.relu = torch.nn.ReLU()
            self.pos_encoder = PositionalEncoding(1244, dropout)
            self.self_att_pool = SelfAttentionPooling(d_model)
            encoder_layers = TransformerEncoderLayer(d_model=d_model, 
                                                    nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout,
                                                    batch_first=True
                                                    )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
            self.d_model = d_model
            self.flatten_layer = torch.nn.Flatten()
            self.decoder = nn.Sequential(nn.Linear(d_model, d_model), 
                                        nn.Dropout(dropout_other),
                                        nn.Linear(d_model, d_model), 
                                        nn.Linear(d_model, 64))
            self.fc_out1 = torch.nn.Linear(64, 64)
            self.fc_out2 = torch.nn.Linear(64, 1) # fix: atm forced binary output layer
            # Transformer Conv. layers
            self.conv1 = torch.nn.Conv1d(in_channels=12, out_channels=128, kernel_size=17, stride=1, padding=0)
            self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=d_model, kernel_size=3, stride=1, padding=1)
            self.conv = torch.nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding=0)
            self.maxpool = torch.nn.MaxPool1d(kernel_size=2)
            self.dropout = torch.nn.Dropout(p=0.1)

        def init_weights(self):
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, src):      
            src = self.relu(self.conv1(src))
            src = self.relu(self.conv2(src))
            
            for _ in range(self.n_conv_layers):
                src = self.relu(self.conv(src))
                src = self.maxpool(src)

                src = self.pos_encoder(src)   
                src = src.permute(0,2,1)
                output = self.transformer_encoder(src)
                output = self.self_att_pool(output)
                logits = self.decoder(output) # output: [batch, n_class]
                xc = self.flatten_layer(logits)
                xc = self.fc_out2(self.dropout(self.relu(self.fc_out1(xc)))) 

            return xc
    
    return TransformerModel
