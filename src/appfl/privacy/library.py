# ## TODO: residual calculation + adaptive penalty

# self.residual['primal'] = 0; self.residual['dual'] = 0        
# if t == 0:
#     for i in range(self.num_clients):
#         self.local_state[i] = copy.deepcopy(self.primal_states[i])
# else:
    


#     # adaptive penalty parameter
#     mu = 10
#     tau_incr = 2
#     tau_decr = 2
#     for i in range(self.num_clients):
        
#         if self.residual['primal'] > mu * self.residual['dual']:
#             penalty[i] = penalty[i] * tau_incr
        
#         elif self.residual['dual'] > mu * self.residual['primal']:
#             penalty[i] = penalty[i] / tau_decr
        
#         else:
#             penalty[i] = penalty[i]


# for rank in range(1,comm_size):            
#     for _, cid in enumerate(num_client_groups[rank-1]):
#         model_info[rank]['penalty'][cid] = copy.deepcopy(penalty[cid])


# print("primal_res=", self.residual['primal'], "  dual_res=", self.residual['dual'], " penalty=", penalty[0])














# if self.clip_value != "inf":                           
#     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value, norm_type=self.clip_norm)                

                # grads=[]
                # for param in self.model.parameters():
                #     grads.append(param.grad.view(-1))
                # grads = torch.cat(grads)                     
                # print("grads=", grads, " shape=", grads.shape)

                # c_bar = 1.0     
                # if self.clip_value != "inf":      
                #     c_bar = torch.norm(grads, p=self.clip_norm).item() / self.clip_value                    
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value, norm_type=self.clip_norm)                                                        


            ## Parameter Clipping  
            # self.clip_value = 15000
            # params=[]
            # for name, param in self.model.named_parameters():
            #     params.append( self.local_state[name].view(-1) )
            # params = torch.cat(params)
            # c_bar = torch.norm(params, p=self.clip_norm).item() / self.clip_value  
            # c_bar = max(0.5, c_bar)            
            # for name, param in self.model.named_parameters():
            #     self.local_state[name] = self.local_state[name] / c_bar

            # params=[]
            # for name, param in self.model.named_parameters():
            #     params.append( self.local_state[name].view(-1) )
            # params = torch.cat(params)                

            # print("norm=", torch.norm(params, p=self.clip_norm).item())
    
        # print("c_bar=", c_bar)        
        # if self.epsilon != "inf":                                                
        #     Sensitivity_value = Calculate_Sensitivity()
        #     # print("Sensitivity_value=", Sensitivity_value)            
        #     for name, param in self.model.named_parameters():                       
        #         mean  = torch.zeros_like(param.data)
        #         scale = torch.zeros_like(param.data) + ( Sensitivity_value / self.epsilon )
        #         m = torch.distributions.laplace.Laplace( mean, scale )                    
        #         self.local_grad[name] += m.sample()              