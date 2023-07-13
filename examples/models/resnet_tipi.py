def get_model():
    import torch
    import torchvision
    import torch.nn as nn
    from copy import deepcopy
    from torch.cuda.amp import autocast, GradScaler
    from torchvision.models.feature_extraction import create_feature_extractor
    scaler = GradScaler()
    scaler_adv = GradScaler()

    class TIPI(nn.Module):
        def __init__(self, num_output, lr_per_sample=0.00025/64, optim='SGD', epsilon=2/255, random_init_adv=False, reverse_kl=True,  tent_coeff=0.0, use_test_bn_with_large_batches=False):
            super(TIPI, self).__init__()

            self.lr_per_sample = lr_per_sample
            self.epsilon = epsilon
            self.random_init_adv = random_init_adv
            self.reverse_kl = reverse_kl
            self.tent_coeff = tent_coeff
            self.use_test_bn_with_large_batches = use_test_bn_with_large_batches
            self.large_batch_threshold = 64

            self.ResNet18 =  torchvision.models.resnet18(pretrained=True)
            self.ResNet18.fc = nn.Sequential(nn.Linear(512, num_output))
            self.need_setup = True
            self.optim = optim
            self.lr_per_sample = lr_per_sample
            

        def __set_up_model(self):  

            configure_multiple_BN(self.ResNet18,["main","adv"])  
            params, _ = collect_params(self.ResNet18)

            if self.optim == 'SGD':
                self.optimizer = torch.optim.SGD(params, lr=self.lr_per_sample,
                                            momentum=0.9,
                                            weight_decay=0.0)
            elif self.optim == 'Adam':
                self.optimizer = torch.optim.Adam(params, lr=self.lr_per_sample,
                                            betas=(0.9, 0.999),
                                            weight_decay=0.0)
            else:
                raise NotImplementedError
            # Don't need to set up again
            self.need_setup = False

        def forward(self, x):
            if self.need_setup:
                self.__set_up_model()

            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_per_sample * x.shape[0]
            
            with autocast():
                self.ResNet18.train()
                use_BN_layer(self.ResNet18,'main')

                delta = torch.zeros_like(x)
                delta.requires_grad_()
                torch.set_grad_enabled(True)
                pred = self.ResNet18(x+delta)
            
                use_BN_layer(self.ResNet18,'adv')
                if self.random_init_adv:
                    delta = (torch.rand_like(x)*2-1) * self.epsilon
                    delta.requires_grad_()
                    pred_adv = self.ResNet18(x+delta)
                else:
                    pred_adv = pred
                
                loss = KL(pred.detach(), pred_adv, reverse=self.reverse_kl).mean()
                grad = torch.autograd.grad(scaler_adv.scale(loss), [delta], retain_graph=(self.tent_coeff!=0.0) and (not self.random_init_adv))[0]
                delta = delta.detach() + self.epsilon*torch.sign(grad.detach())
                delta = torch.clip(delta,-self.epsilon,self.epsilon)
                x_adv = x + delta
                #x_adv = torch.clip(x_adv,0.0,1.0)

                
                pred_adv = self.ResNet18(x_adv)
                loss = KL(pred.detach(), pred_adv, reverse=self.reverse_kl)
                ent = - (pred.softmax(1) * pred.log_softmax(1)).sum(1)
                
                if self.tent_coeff != 0.0:
                    loss = loss + self.tent_coeff*ent
                
                loss = loss.mean()
                
                self.optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                #scaler_adv.update()
                #self.optimizer.step()

                
                use_BN_layer(self.ResNet18,'main')
                
                with torch.no_grad():
                    if self.use_test_bn_with_large_batches and x.shape[0] > self.large_batch_threshold:
                        pred = self.ResNet18(x)
                    else:
                        self.ResNet18.eval()
                        pred = self.ResNet18(x)
               
                return pred


    class MultiBatchNorm2d(nn.Module):
        def __init__(self, bn, BN_layers=['main']):
            super(MultiBatchNorm2d, self).__init__()
            self.weight = bn.weight
            self.bias = bn.bias
            self.BNs = nn.ModuleDict()
            self.current_layer = 'main'
            for l in BN_layers:
                m = deepcopy(bn)
                m.weight = self.weight
                m.bias = self.bias
                try:
                    self.BNs[l] = m
                except Exception:
                    import pdb; pdb.set_trace()
        def forward(self,x):
            assert self.current_layer in self.BNs.keys()
            return self.BNs[self.current_layer](x)

    def collect_params(model):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, MultiBatchNorm2d)\
                    or isinstance(m,nn.GroupNorm)\
                    or isinstance(m,nn.InstanceNorm2d)\
                    or isinstance(m,nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_multiple_BN(net, BN_layers=['main']):
        for attr_str in dir(net):
            m = getattr(net, attr_str)
            if type(m) == nn.BatchNorm2d:
                new_bn = MultiBatchNorm2d(m, BN_layers)
                setattr(net, attr_str, new_bn)
        for n, ch in net.named_children():
            if type(ch) != MultiBatchNorm2d:
                configure_multiple_BN(ch, BN_layers)

    def use_BN_layer(net, BN_layer='main'):
        for m in net.modules():
            if isinstance(m, MultiBatchNorm2d):
                m.current_layer = BN_layer



    def KL(logit1,logit2,reverse=False):
        if reverse:
            logit1, logit2 = logit2, logit1
        p1 = logit1.softmax(1)
        logp1 = logit1.log_softmax(1)
        logp2 = logit2.log_softmax(1) 
        return (p1*(logp1-logp2)).sum(1)

    return TIPI