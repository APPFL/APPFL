from .server_federated_bfgs import FedServerBFGS
import torch
import copy

class ServerFedBFGSPCA(FedServerBFGS):

    def update_global_state(self, clients):

        super(ServerFedBFGSPCA, self).compute_pseudo_gradient()

        
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
        

        step_size = 1.0
        if self.round > 0:     
            step_size = 1.0            

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
                global_state_vec_next = self.global_state_vec + torch.mm( self.P.transpose(0, 1), step_size * direction )
    

                self.update_model_param(model, global_state_vec_next)            

                ## compute loss_new        
                loss_new = 0
                for k, client in enumerate(clients):            
                    loss_new += self.loss_calculation(client.loss_type, model[k], client.dataloader)
                loss_new = loss_new / self.num_clients
 
                if loss_new <= RHS or step_size <= 1e-10:
                    termination = 0
                else:
                    step_size = step_size * self.tau

        print("step_size=", step_size)

        ## compute step   
        step = step_size * direction

        ## update global state
        self.global_state_vec += torch.mm( self.P.transpose(0, 1), step )

        ## store
        self.round += 1        
        self.step_prev = copy.deepcopy(step)
        self.pseudo_grad_prev = copy.deepcopy(self.pseudo_grad)

    def update_model_param(self, model, vector):
        for k in range(self.num_clients):
            idx = 0
            for _,param in model[k].named_parameters():
                arr_shape = param.data.shape
                size = 1
                for i in range(len(list(arr_shape))):
                    size *= arr_shape[i]
                param.data = vector[idx:idx+size].reshape(arr_shape)
                idx += size  


    def loss_calculation(self, loss_type, model, dataloader):
        device = self.device
 
        loss_fn = eval(loss_type)
         
        model.to(device)
        model.eval()
        loss = 0
        tmpcnt = 0 
        with torch.no_grad():
            for img, target in dataloader:
                tmpcnt += 1 
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                loss +=  loss_fn(output, target).item()

        loss = loss / tmpcnt        

        return loss 

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
