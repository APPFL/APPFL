from .Server_fed import FedServer

class ServerFedAvgMomentum(FedServer):
    def compute_step(self):        
        super(ServerFedAvgMomentum, self).compute_pseudo_gradient()
        super(ServerFedAvgMomentum, self).update_m_vector()
        for name, _ in self.model.named_parameters():                        
            self.step[name] = self.m_vector[name]