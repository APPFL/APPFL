from typing import OrderedDict
from .server_federated import FedServer
import torch

class ServerFedBFGS(FedServer):
    def compute_step(self):                
        super(ServerFedBFGS, self).compute_pseudo_gradient()

        
        for name, _ in self.model.named_parameters():

            print("org=", self.pseudo_grad["conv1.weight"] )
            
            self.pseudo_grad_vec[name] = torch.flatten(self.pseudo_grad[name])
            print("size=", self.pseudo_grad_vec[name].size() )

            self.pseudo_grad_vec[name] += 1.0


            self.pseudo_grad[name] = torch.reshape( self.pseudo_grad_vec[name], self.model_size[name] )


            print("mod=", self.pseudo_grad["conv1.weight"] )
 
            
            # print("org=", self.pseudo_grad[name].size() )
            # print("fla=", torch.flatten(self.pseudo_grad[name]).size() )


            self.step[name] = 0
        
