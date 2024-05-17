import math 
import torch 
import json 
import logging as log 
import torch.nn as nn 
import torch.nn.functional as F 

class RNN_default(nn.Module):
    def __init__(self,
                 cell_type,
                 input_size,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 dropout,
                 proj,
                 init_type,
                 ):
        super(RNN_default,self).__init__()

        self.cell=cell_type
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.dropout=dropout
        self.proj=proj
        self.init_type=init_type

        if cell_type=="lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif cell_type=="gru":
                self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            raise ValueError("Invalid RNN cell type")
        
        if proj is not None:
            self.output_size=proj
            self.linear=nn.Linear(hidden_size*2 if bidirectional else hidden_size,proj)
        else:
            self.output_size=hidden_size*2 if bidirectional else hidden_size
    
    def reset_parameters(self):
        if self.init_type == "kaiming_uniform":
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        elif self.init_type == "kaiming_normal":
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.kaiming_normal_(param, a=0, mode="fan_in")
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        elif self.init_type == "xavier_uniform":
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        elif self.init_type == "xavier_normal":
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_normal_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
        else:
            raise ValueError(f"Invalid initialization type: {self.init_type}")
    
    def forward(self,x,lens=None):

        if lens is not None:
            x=nn.utils.rnn.pack_padded_sequence(x,lens,batch_first=True)
        
        rnn_out,_=self.rnn(x)

        if lens is not None:
            rnn_out,_=nn.utils.rnn.pad_packed_sequence(rnn_out,batch_first=True)
        if self.bidirectional:
            rnn_out=torch.cat((rnn_out[:,:,:self.hidden_size],rnn_out[:,:,self.hidden_size:]),dim=2)
        if self.proj is not None:
            rnn_out=self.linear(rnn_out)
        
        return rnn_out, lens 

class Linear(nn.Module):
    def __init__(self, input_size, output_size,init_type):
        super(Linear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.init_type=init_type

        self.linear = nn.Linear(input_size, output_size)
    
    def reset_parameters(self):
        if self.init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
        elif self.init_type == "kaiming_normal":
            nn.init.kaiming_normal_(self.weight, a=0, mode="fan_in")
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif self.init_type == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif self.init_type == "xavier_normal":
            nn.init.xavier_normal_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else:
            raise ValueError(f"Invalid initialization type: {self.init_type}")

    def forward(self, x):
        return self.linear(x)

class MultiViewRNN(nn.Module):
    def __init__(self,config_file):

        nn.Module.__init__(self)

        with open(config_file,"r") as f:
            config=json.load(f)

        self.net=nn.ModuleDict()
        self.init_type=config["init_type"]

        log.info(f"view1:")
        view1_config=config["view1"]
        self.net["view1"]=RNN_default(cell_type=view1_config["cell_type"],
                                      input_size=view1_config["input_size"],
                                      hidden_size=view1_config["hidden_size"],
                                      num_layers=view1_config["num_layers"],
                                      bidirectional=view1_config["bidirectional"],
                                      dropout=view1_config["dropout"],
                                      proj=view1_config["proj"],
                                      init_type=self.init_type
                                      )
        
        log.info(f"view2:")
        view2_config=config["view2"]
        self.net["view2"]=RNN_default(cell_type=view2_config["cell_type"],
                                      input_size=view2_config["input_size"],
                                      hidden_size=view2_config["hidden_size"],
                                      num_layers=view2_config["num_layers"],
                                      bidirectional=view2_config["bidirectional"],
                                      dropout=view2_config["dropout"],
                                      proj=view2_config["proj"],
                                      init_type=self.init_type
                                      )
        
        if "proj" in config and config["proj"] is not None:

            log.info(f"proj:")
            self.net["proj"]=Linear(self.net["view1"].output_size,config["proj"],init_type=self.init_type)

        if "loss_fn" in config and config["loss_fn"] is not None:
            self.loss_fn=config["loss_fn"]
    
    @property
    def output_size(self):
        if "proj" in self.net:
            return self.net["proj"].output_size
        else:
            return self.net["view1"].output_size
         
    def forward(self,batch):
        view1_in=batch["view1"]
        view2_in=batch["view2"]

        view1_out,_=self.net["view1"](view1_in)
        view2_out,_=self.net["view2"](view2_in)

        if "proj" in self.net:
            view1_out=self.net["proj"](view1_out)
            view2_out=self.net["proj"][view2_out]


        return view1_out,view1_out

if __name__=="__main__":

    model=MultiViewRNN("config.json")

    print(model)
