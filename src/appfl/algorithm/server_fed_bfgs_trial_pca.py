from .server_federated_bfgs import FedServerBFGS
import torch
import copy

class ServerFedBFGSTrialPCA(FedServerBFGS):

    def update_global_state(self, clients):

        super(ServerFedBFGSTrialPCA, self).compute_pseudo_gradient()
        
        if self.round == 0:
            
            self.global_state_vec += - torch.mm( self.P.transpose(0, 1), self.pseudo_grad )

        else: 
            ## compute the approximate inverse Hessian
            if self.round > 1:
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
            

            step_size = 1.0
            ## backtracking line search
            if self.round > 1:     
                step_size = 1.0        
                backtracking_line_search()    

                ## deepcopy the models        
                model = {}                
                for k, client in enumerate(clients):                        
                    model[k] = copy.deepcopy(client.model)             
                    
                ## update model     
                self.update_model_param(model, self.global_state_vec)            
                
                ## compute loss_prev        
                loss_prev = 0
                for k, client in enumerate(clients):            
                    loss_prev += self.loss_calculation(client.loss_type, model[k], client.dataloader)
                loss_prev = loss_prev / self.num_clients
                
                termination = 1
                while termination:
                    
                    ##  
                    RHS = loss_prev + self.c * step_size * torch.dot(self.pseudo_grad.reshape(-1), direction.reshape(-1))

                    ##
                    global_state_vec_next = self.global_state_vec + torch.mm( self.P.transpose(0, 1), step_size * direction - self.pseudo_grad )
        

                    self.update_model_param(model, global_state_vec_next)            

                    ## compute loss_new        
                    loss_new = 0
                    for k, client in enumerate(clients):            
                        loss_new += self.loss_calculation(client.loss_type, model[k], client.dataloader)
                    loss_new = loss_new / self.num_clients

                    # print("loss_new=", loss_new, " RHS=", RHS, " step_size=", step_size)


                    if loss_new <= RHS or step_size <= 1e-10:
                        termination = 0
                    else:
                        step_size = step_size * self.tau

            print("step_size=", step_size)

            ## compute step           
            step = step_size * direction - self.pseudo_grad
        

            ## update global state
            self.global_state_vec += torch.mm( self.P.transpose(0, 1), step )

            ## store        
            self.step_prev = copy.deepcopy(step)
            self.pseudo_grad_prev = copy.deepcopy(self.pseudo_grad)


        self.round += 1        

    

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
