import copy

from collections import OrderedDict
import torch


"""This implements a base class for server."""


class BaseServer:
    def __init__(self, weights, model, num_clients, device):
        self.model = model
        self.num_clients = num_clients
        self.device = device
        self.weights = weights    
        self.penalty = OrderedDict()
        self.primal_states = OrderedDict()
        self.dual_states = OrderedDict()
        self.primal_states_curr = OrderedDict()
        self.primal_states_prev = OrderedDict()
        for i in range(num_clients):
            self.primal_states[i] = OrderedDict()
            self.dual_states[i] = OrderedDict()
            self.primal_states_curr[i] = OrderedDict()
            self.primal_states_prev[i] = OrderedDict()

    # update global model
    def update(self):
        raise NotImplementedError

    def get_model(self):
        return copy.deepcopy(self.model)
    
    def primal_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):            
            if states is not None:
                for sid, state in states.items():
                    self.primal_states[sid] = copy.deepcopy(state["primal"])

    def dual_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):
            if states is not None:
                for sid, state in states.items():
                    self.dual_states[sid] = copy.deepcopy(state["dual"])

    def penalty_recover_from_local_states(self, local_states):
        for _, states in enumerate(local_states):
            if states is not None:
                for sid, state in states.items():                    
                    self.penalty[sid] = copy.deepcopy(state["penalty"][sid])     
    
    def primal_residual_at_server(self, global_state):
        primal_res = 0
        for i in range(self.num_clients):
            for name, param in self.model.named_parameters():
                primal_res += torch.sum(  torch.square(  global_state[name] - self.primal_states[i][name].to(self.device)  )  ) 
        primal_res = torch.sqrt(primal_res).item()          
        return primal_res

    def dual_residual_at_server(self):
        dual_res = 0
        if self.is_first_iter == 1:
            for i in range(self.num_clients):
                for name, _ in self.model.named_parameters():
                    self.primal_states_curr[i][name] = copy.deepcopy( self.primal_states[i][name].to(self.device) )
            self.is_first_iter = 0
        
        else:
            self.primal_states_prev = copy.deepcopy( self.primal_states_curr )
            for i in range(self.num_clients):
                for name, _ in self.model.named_parameters():
                    self.primal_states_curr[i][name] = copy.deepcopy( self.primal_states[i][name].to(self.device) )
            
            ## compute dual residual
            for name, _ in self.model.named_parameters():
                temp = 0
                for i in range(self.num_clients):
                    temp += self.penalty[i] * (  self.primal_states_prev[i][name] - self.primal_states_curr[i][name])
                
                dual_res += torch.sum( torch.square( temp ) )
            dual_res = torch.sqrt(dual_res).item()          
                            
        return dual_res

         

"""This implements a base class for clients."""


class BaseClient:
    def __init__(self, id, weight, model, dataloader, device):
        self.id = id
        self.weight = weight
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.primal_state = OrderedDict()
        self.dual_state = OrderedDict()

    # update local model
    def update(self):
        raise NotImplementedError

    def get_model(self):
        return self.model.state_dict()


    """ 
    Differential Privacy 
        (Laplacian mechanism) 
        - Noises from a Laplace dist. with zero mean and "scale_value" are added to the primal_state 
        - Variance = 2*(scale_value)^2
        - scale_value = sensitivty/epsilon, where sensitivity is determined by data, algorithm.
    """        

    def laplace_mechanism_output_perturb(self,scale_value):       

        for name, param in self.model.named_parameters():
            mean = torch.zeros_like(param.data)
            scale = torch.zeros_like(param.data) + scale_value
            m = torch.distributions.laplace.Laplace(mean, scale)
            self.primal_state[name] += m.sample()
