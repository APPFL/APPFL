 # @torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean(0)

def get_model():
    from copy import deepcopy
    import torch
    import torch.nn as nn
    import torch.jit
    import torchvision
    import torch.optim as optim

    def copy_model_and_optimizer(model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())
        return model_state, optimizer_state


    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """Restore the model and optimizer states from copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
    
    def prepare_bn_layer(bn_layer, is_training, bn_mode):
        if bn_mode == "fixed":
            if is_training:
                bn_layer.requires_grad_(True)
                bn_layer.track_running_stats = False
            else:
                bn_layer.eval()
        elif bn_mode == "adaptation":
            if is_training:
                bn_layer.requires_grad_(True)
                bn_layer.train()
            else:
                bn_layer.eval()
        elif bn_mode == "none":
            if is_training:
                bn_layer.requires_grad_(True)

            bn_layer.track_running_stats = False
            bn_layer.running_mean = None
            bn_layer.running_var = None
        else:
            assert NotImplementedError
        # bn_layer.eval()
        
        # return bn_layer
    
    class ResNet(nn.Module):
        """ResNet18 model with additional Sigmoid layer for classification
        """

        def __init__(self, num_output):
            super(ResNet, self).__init__()
            self.ResNet18 = torchvision.models.resnet18(pretrained=True)
            self.ResNet18.fc = nn.Sequential(nn.Linear(512, num_output))

        def forward(self, x):
            x = self.ResNet18(x)
            return x

    class Tent(nn.Module):
        """Tent adapts a model by entropy minimization during testing.

        Once tented, a model adapts itself by updating on every forward.
        """
        def __init__(self, num_output, lr, update_bn=True, update_fc=True, momentum=0.9, dampening=0.0, wd=0.0, nesterov = True, 
                     steps=1, bn_mode="fixed", unsupervised=True):
            super().__init__()
            self.ResNet18 = torchvision.models.resnet18(pretrained=True)
            self.ResNet18.fc = nn.Sequential(nn.Linear(512, num_output))

            self.optimizer = None
            self.lr = lr
            self.momentum = momentum
            self.dampening = dampening
            self.wd = wd
            self.nesterov = nesterov
            self.steps = steps
            assert steps > 0, "tent requires >= 1 step(s) to forward and update"
            self.need_setup = True
            self.update_bn = update_bn
            self.update_fc = update_fc
            assert bn_mode in ["fixed", "adaptation", "none"]
            self.bn_mode = bn_mode
            self.unsupervisied = unsupervised
        
        def forward_and_adapt(x, model, optimizer):
            """Forward and adapt model on batch of data.

            Measure entropy of the model prediction, take gradients, and update params.
            """
            with torch.enable_grad():
                # forward
                outputs = model(x)
                # adapt
                loss = softmax_entropy(outputs).mean(0)
                # loss = -(x.softmax(1) * x.log_softmax(1)).sum(1).mean(0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            return outputs

        def forward_only(x, model):
            with torch.enable_grad():
                outputs = model(x)
            return outputs

        def get_loss(self):
            return softmax_entropy

        def collect_params(self, model):
            """Collect the affine scale + shift parameters from batch norms.

            Walk the model's modules and collect all batch normalization parameters.
            Return the parameters and their names.

            Note: other choices of parameterization are possible!
            """
            params = []
            names = []
            for nm, m in model.named_modules():
                # if isinstance(m, nn.BatchNorm2d):
                if (isinstance(m, nn.Linear) and self.update_fc) or (isinstance(m, nn.BatchNorm2d) and self.update_bn):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{nm}.{np}")
            return params, names

        def check_model(model):
            """Check model for compatability with tent."""
            is_training = model.training
            assert is_training, "tent needs train mode: call model.train()"
            param_grads = [p.requires_grad for p in model.parameters()]
            has_any_params = any(param_grads)
            has_all_params = all(param_grads)
            assert has_any_params, "tent needs params to update: " \
                                "check which require grad"
            assert not has_all_params, "tent should not update all params: " \
                                    "check which require grad"
            has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
            assert has_bn, "tent needs normalization for its optimization"
        

        def prepare_model_before_training(self, model):
            """Configure model for use with tent."""
            # train mode, because tent optimizes the model to minimize entropy
            model.train()
            # disable grad, to (re-)enable only what tent updates
            model.requires_grad_(False)
            # configure norm for tent updates: enable grad + force batch statisics
            for m in model.modules():
                if (self.update_fc and isinstance(m, nn.Linear)):
                    m.requires_grad_(True)
                if (self.update_bn and isinstance(m, nn.BatchNorm2d)):
                    # Use the statistics from pre-trained model
                    m = prepare_bn_layer(m, is_training=True, bn_mode=self.bn_mode)
            return model

        def prepare_model_before_testing(self):
            for m in self.ResNet18.modules():
                if (self.update_bn and isinstance(m, nn.BatchNorm2d)):
                    m = prepare_bn_layer(m, is_training=False, bn_mode=self.bn_mode)

        def setup_adaptation(self):
            # Configure model
            self.ResNet18 = self.prepare_model_before_training(self.ResNet18)
            # Check model
            Tent.check_model(self.ResNet18)
            # Select params to optimize
            params, params_name = self.collect_params(self.ResNet18)
            # print(params_name)
            self.optimizer = optim.SGD(params,
                lr=self.lr,
                momentum=self.momentum,
                dampening=self.dampening,
                weight_decay=self.wd,
                nesterov=self.nesterov)
            self.need_setup = False
                
        def forward(self, x, perform_adapt=True):
            outputs = self.ResNet18(x)
            return outputs
    return Tent