import torch 
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


num_layers=2
batch_size=32
epochs=20
lr=0.0001
tcher_force=0.5
optimizer='adam'
hidden_dim=512
embed_dim=128
input_dim=50
output_dim=50
pad_index=0


class RNNModel(nn.Module):
    def __init__(self,input_dim,embed_dim,hidden_dim,num_layers):
        super(RNNModel, self).__init__()
        self.embed=nn.Embedding(input_dim,embed_dim)
        # self.rnn=nn.RNN(embed_dim,hidden_dim,num_layers,batch_first=True)#dk what is batch_first suggested by copilot
        self.encoder_rnn=nn.RNN(embed_dim,hidden_dim,num_layers,batch_first=True)
        self.decoder_rnn=nn.RNN(embed_dim+hidden_dim,hidden_dim,num_layers,batch_first=True)
        #for eij
        self.w_s=nn.Linear(hidden_dim,hidden_dim)
        self.w_h=nn.Linear(hidden_dim,hidden_dim)
        self.const_bahdanau=nn.Linear(hidden_dim,1)
        self.out=nn.Linear(hidden_dim,output_dim)

    def encode(self,input,input_size):
        embed_input=self.embed(input)
        packed_input=nn.utils.rnn.pack_padded_sequence(embed_input,input_size.cpu(),batch_first=True,enforce_sorted=False)     
        packed_output,last_hidden=self.encoder_rnn(packed_input) 
        outputs=nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)[0] 
        return outputs,last_hidden
    
    def bahdanau(self,last_hidden,all_hidden,mask):
        last_hidden=last_hidden.unsqueeze(1)
        eij=self.const_bahdanau(torch.tanh(self.w_h(all_hidden)+self.w_s(last_hidden))).squeeze(2) #eij=tanh(W_h*h_j+W_s*s_i)
        eij=eij.masked_fill(mask==0,-1e10)
        alpha_ij=F.softmax(eij,dim=1)
        c_i=torch.bmm(alpha_ij.unsqueeze(1),all_hidden).squeeze(1)
        return c_i
    
    def decode(self,input,all_hidden,state_hidden,mask):
        prev_state_hidden=state_hidden[-1]
        c_i=self.bahdanau(prev_state_hidden,all_hidden,mask)
        embed_input=self.embed(input)#y_(t-1)
        context_added_input=torch.cat((embed_input,c_i),dim=1).unsqueeze(1)#y_(t-1),c_t combined 
        output,state_hidden=self.decoder_rnn(context_added_input,state_hidden)#y_t,s_t
        output=output.squeeze(1)
        pred=self.out(output)
        return pred,state_hidden
    
    
    





