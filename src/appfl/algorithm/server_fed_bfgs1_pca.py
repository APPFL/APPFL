from .server_federated_bfgs import FedServerBFGS
import torch
import copy

class ServerFedBFGS1PCA(FedServerBFGS):

    def update_global_state(self, clients):

        super(ServerFedBFGS1PCA, self).compute_pseudo_gradient()

        
        ## compute the approximate inverse Hessian
        if self.round > 0:            
            s = self.step_prev 
            s = s.reshape(-1)
            y = self.pseudo_grad - self.pseudo_grad_prev
            y = y.reshape(-1)
            rho = 1.0 / torch.dot(y,s)             
            matrix_1 = self.I_matrix - rho * torch.outer(s, y)
            matrix_2 = self.I_matrix - rho * torch.outer(y, s)
            matrix_3 = rho * torch.outer(s, s) 
            temp1 = torch.mm(matrix_1,self.H_matrix)
            self.H_matrix = torch.mm(temp1,matrix_2) + matrix_3
 
        ## compute direction        
        direction = - torch.mm( self.H_matrix, self.pseudo_grad )
        

        step_size = self.server_learning_rate 
        

        ## compute step   
        step = step_size * direction

        ## update global state
        self.global_state_vec += torch.mm( self.P.transpose(0, 1), step )

        ## store
        self.round += 1        
        self.step_prev = copy.deepcopy(step)
        self.pseudo_grad_prev = copy.deepcopy(self.pseudo_grad)

     

    def logging_summary(self, cfg, logger):
        
        super(FedServerBFGS, self).log_summary(cfg, logger)

        logger.info("client_learning_rate = %s " % (cfg.fed.args.optim_args.lr))

        if cfg.summary_file != "":
            with open(cfg.summary_file, "a") as f:

                f.write(
                    cfg.logginginfo.DataSet_name
                    + " FedAvg ClientLR "
                    + str(cfg.fed.args.optim_args.lr)
                    + " TestAccuracy "
                    + str(cfg.logginginfo.accuracy)
                    + " BestAccuracy "
                    + str(cfg.logginginfo.BestAccuracy)
                    + " Time "
                    + str(round(cfg.logginginfo.Elapsed_time, 2))
                    + "\n"
                )
