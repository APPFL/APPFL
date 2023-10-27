import torch
from torch import nn
from torch.distributions.laplace import Laplace
from typing import List


class AttentionLSTM(nn.Module):
    
    def __init__(self, args):
        """
        Initialize the network.

        Args:
            args: (ParserArgs): Parser arguments
        ParserArgs:
            input_size: (int): size of the input
            hidden_size: (int): hidden size of encoder states
            seq_len: (int): sequence length
            num_lstm_layers: (int): Number of layers in the lstm
        """
        
        super(AttentionLSTM, self).__init__()
        
        # extract parameters from args
        self.input_size = args.input_size # args
        self.hidden_size = args.hidden_size # args
        self.seq_len = args.seq_len # args
        self.num_lstm_layers = args.num_lstm_layers # args
            
        # not currently supporting single-feature inputs
        if self.input_size < 2:
            raise ValueError('Currently not supporting sigle feature inputs for past data points.')
        else:
            self.encoder_in = self.input_size-1
            self.decoder_in = 1
        
        # define local submodules    
        self.lstm = nn.LSTM(
            input_size=self.encoder_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first = True
        )
        
        self.lstm_d = nn.LSTM(
            input_size=self.hidden_size+1,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first = True
        )

        self.attn = nn.Sequential( 
            nn.Linear(in_features=2*self.hidden_size+self.seq_len,out_features=self.encoder_in),
            nn.Tanh(),
            nn.Linear(in_features=self.encoder_in,out_features=1,bias=False),
            nn.Flatten(start_dim=-2,end_dim=-1),
            nn.Softmax(dim=-1)
        )
        
        self.attn_d = nn.Sequential(
            nn.Linear(in_features=3*self.hidden_size,out_features=self.hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=self.hidden_size,out_features=1,bias=False),
            nn.Flatten(start_dim=-2,end_dim=-1),
            nn.Softmax(dim=-1)
        )
        
        self.proj_output = nn.Linear(in_features=2*self.hidden_size,out_features=1)

    def forward(self, input_data: torch.Tensor, enable_attention: bool = True, out_attention: bool = False):
        """
        Forward computation.

        Args:
            enable_attention: (bool): True if attention weights are calculated else False
        """
        input_encoder = input_data[:,:,1:]
        input_decoder = input_data[:,:,0]
        
        h = torch.zeros(self.num_lstm_layers, input_encoder.size(0), self.hidden_size, device=input_encoder.device)
        c = torch.zeros(self.num_lstm_layers, input_encoder.size(0), self.hidden_size, device=input_encoder.device)
        h_d = torch.zeros(self.num_lstm_layers, input_decoder.size(0), self.hidden_size, device = input_decoder.device)
        c_d = torch.zeros(self.num_lstm_layers, input_decoder.size(0), self.hidden_size, device = input_decoder.device)
        attentions = torch.zeros(input_data.size(0), self.seq_len, self.encoder_in, device=input_data.device)
        attentions_d = torch.zeros(input_data.size(0), self.seq_len, self.seq_len, device=input_data.device)
        
        top_lstm_state_collector = []
            
        # save permuted version of input if attentions need to be calculated
        if enable_attention:
            input_permuted = input_encoder.permute(0,2,1).to(input_data.device)

        # encoder
        for t in range(self.seq_len):
            
            if enable_attention:
            
                # concatenate input to attention layers
                x = torch.cat((h[-1,:,:][:,None,:].repeat(1,self.encoder_in,1),
                    c[-1,:,:][:,None,:].repeat(1,self.encoder_in,1),
                    input_permuted),dim=-1).to(input_encoder.device)
                
                # get attention weights
                a_t = self.attn(x)

                # record attentions
                attentions[:, t, :] = a_t
                
                # move lstm one step forward
                this_input = input_encoder[:,[t],:]
                _,(h,c) = self.lstm(torch.mul(this_input,a_t.view_as(this_input)),(h,c))
                
            else:
                
                # move lstm one step forward
                this_input = input_encoder[:,[t],:]
                _,(h,c) = self.lstm(torch.mul(this_input,1/(self.encoder_in)),(h,c))
            
            # append lstm state
            top_lstm_state_collector.append(h[-1,:,:])
        
        # decoder
        h_expanded = torch.cat([x[:,None,:] for x in top_lstm_state_collector],dim=1).to(input_decoder.device)
        for t in range(self.seq_len):
            
            if enable_attention:
                
                # concatenate input to attention layers
                x = torch.cat([h_d[-1,:,:][:,None,:].repeat(1,self.seq_len,1),
                        c_d[-1,:,:][:,None,:].repeat(1,self.seq_len,1),
                        h_expanded],dim=-1).to(input_decoder.device)
                
                # get attention weights
                a_t_d = self.attn_d(x)
                
                # record attentions
                attentions_d[:, t, :] = a_t_d
                
                # move lstm one step forward
                partial_input = torch.sum(torch.mul(h_expanded,a_t_d[:,:,None].repeat(1,1,self.hidden_size)),dim=1)
                full_input = torch.cat([partial_input,input_decoder[:,[t]]],dim=-1).to(input_decoder.device)
                _,(h_d,c_d) = self.lstm_d(full_input[:,None,:],(h_d,c_d))
                
            else:
                
                # move lstm one step forward
                partial_input = torch.sum(torch.mul(h_expanded,1/self.seq_len),dim=1)
                full_input = torch.cat([partial_input,input_decoder[:,[t]]],dim=-1).to(input_decoder.device)
                _,(h_d,c_d) = self.lstm_d(full_input[:,None,:],(h_d,c_d))
                
        out =  self.proj_output(torch.cat([h_d[-1,:,:],partial_input],dim=1)).abs()
        
        if out_attention:
            return out,attentions,attentions_d       
        else: 
            return out
    
    def to(self, device):

        self.lstm = self.lstm.to(device)
        self.lstm_d = self.lstm_d.to(device)
        self.attn = self.attn.to(device)
        self.attn_d = self.attn_d.to(device)
        self.proj_output = self.proj_output.to(device)
        return super(AttentionLSTM,self).to(device)