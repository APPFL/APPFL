from .Server_fed import FedAdaptServer

class FedAvgMServer(FedAdaptServer):
    def compute_step(self):        
        super(FedAvgMServer, self).update_m_vector()
        for name, _ in self.model.named_parameters():                        
            self.step[name] = self.m_vector[name]