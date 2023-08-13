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
    from torch.nn import functional as F
    from sklearn import metrics
    import numpy as np
    import copy

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

    class TentAdaptTest(nn.Module):
        """Tent adapts a model by entropy minimization during testing.

        Once tented, a model adapts itself by updating on every forward.
        """
        def __init__(self, num_output, lr, update_bn=True, update_fc=True, momentum=0.9, dampening=0.0, wd=0.0, nesterov = True, 
                     adapt_steps=1, test_steps=1, adapt_batchsize=200, test_batchsize=200,  bn_mode="fixed", unsupervised=True,
                     adapt_at_test=True
                     ):
            super().__init__()
            self.ResNet18 = torchvision.models.resnet18(pretrained=True)
            self.ResNet18.fc = nn.Sequential(nn.Linear(512, num_output))
            
            # Record optimizer's parameters
            self.optimizer = None
            self.lr = lr
            self.momentum = momentum
            self.dampening = dampening
            self.wd = wd
            self.nesterov = nesterov
            
            # Record training/testing parameters
            self.adapt_batchsize = adapt_batchsize
            self.test_batchsize = test_batchsize
            self.adapt_at_test = adapt_at_test

            assert (adapt_steps > 0 and test_steps > 0), "tent requires >= 1 step(s) to forward and update"
            self.adapt_steps = adapt_steps
            self.test_steps = test_steps

            self.adapt_at_test = adapt_at_test
            self.need_setup = True
            self.update_bn = update_bn
            self.update_fc = update_fc
            assert bn_mode in ["fixed", "adaptation", "none"]
            self.bn_mode = bn_mode
            self.unsupervisied = unsupervised

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
            # disable grad, to (re-)enable only what tent will update
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
            TentAdaptTest.check_model(self.ResNet18)
            # Select params to optimize
            params, params_name = self.collect_params(self.ResNet18)
            self.need_setup = False
            return params

        def adaptation_step(self, adaptset_dataloader, device, loss_fn = None):
            # Passing model to the target device
            self.ResNet18.to(device)
            self.ResNet18.eval()
            adapt_log = {}
            
            unsupervised = self.unsupervisied
            ## get loss function
            if unsupervised == True:
                loss_fn = self.get_loss()

            ## prepare model, optimizer for adaptation
            params = self.setup_adaptation()
            optimizer = optim.SGD(params,
                lr=self.lr,
                momentum=self.momentum,
                dampening=self.dampening,
                weight_decay=self.wd,
                nesterov=self.nesterov)
            
            ## local training
            for t in range(self.adapt_steps):
                train_loss = 0
                tmptotal = 0
                targets = []
                preds = []

                for data, target in adaptset_dataloader:
                    data = data.to(device)
                    target = target.to(device)
                    optimizer.zero_grad()
                    
                    output = self.ResNet18(data)
                    if output.shape[1] == 1:
                        pred = torch.round(output)
                    else:
                        pred =  F.softmax(output, dim=1)
                    
                    targets.append(target.cpu().detach().numpy())
                    preds.append(pred.cpu().detach().numpy())
                    
                    if unsupervised == True:
                        loss = loss_fn(output)
                    else:
                        loss = loss_fn(output, target)
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                targets = np.concatenate(targets)
                preds   = np.concatenate(preds)

                # log eval results
                train_loss = train_loss / len(adaptset_dataloader)
                auc = metrics.roc_auc_score(targets, preds[:, 1])
                adapt_log["Adpt%d_AUC" % t] = auc
                adapt_log["Adapt%d_LSS" % t] = train_loss
                print("Adapt epoch %d - loss : %.02f - AUC: %.02f" % (t, train_loss, auc))

            state_dict, loss, eval_dict = self.run_evaluation(adaptset_dataloader, device=device)
            return state_dict, loss, {**adapt_log, **eval_dict}
            
        def testing_step(self, testset_dataloader, adaptset_dataloader, device):
            """
                Perform testing step
                :testset_dataloader: 
            """
            if self.adapt_at_test:
                state_dict, loss, eval_dict = self.run_adaptation_and_evaluation(testset_dataloader=testset_dataloader,
                                                                             adaptset_dataloader=adaptset_dataloader,
                                                                             device=device)
            else:
                state_dict, loss, eval_dict = self.run_evaluation(testset_dataloader, device=device) 
            
            return state_dict, loss, eval_dict

        def run_evaluation(self, eval_dataloader, device):
            """
                Put the model in evaluation mode, then perform evaluation on the eval dataset  
            """
            loss = 0
            correct = 0
            tmpcnt = 0
            tmptotal = 0
            preds   = []
            targets = []
            outputs = []

            with torch.no_grad():
                for img, target in eval_dataloader:
                    tmpcnt += 1
                    tmptotal += len(target)
                    img     = img.to(device)
                    target  = target.to(device)
                    output  = self.ResNet18(img)
                    if output.shape[1] == 1:
                        pred = torch.round(output)
                    else:
                        pred = F.softmax(output, dim=1)
                        
                    preds.append(pred.cpu().detach().numpy())
                    targets.append(target.cpu().detach().numpy())
                    outputs.append(output.cpu().detach().numpy())

            targets = np.concatenate(targets)
            preds   = np.concatenate(preds)
            outputs = np.concatenate(outputs)
            outputs = [outputs[i].tolist() for i in range(len(outputs))]
            preds_binary = preds.argmax(axis=1) 
            acc = (preds_binary == targets).mean()
            # Compute precision, recall, AUC, AP for binary classification
            # Plot the ROC
            fpr, tpr, _ = metrics.roc_curve(targets,   preds[:,1])
            # Plot the Recall-Precision Curve
            arr_precs, arr_recalls, threshold = metrics.precision_recall_curve(targets, preds[:,1])
            prec, rec, f1, sprt  = metrics.precision_recall_fscore_support(targets, preds_binary, average="binary")
            try:
                auc = metrics.roc_auc_score(targets, preds[:, 1])
            except:
                auc = None
            ap  = metrics.average_precision_score(targets, preds[:,1])
            return copy.deepcopy(self.state_dict()), loss, {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "ap": ap, 
                "tpr": tpr.tolist(), "fpr" : fpr.tolist(), "arr_precs": arr_precs.tolist(), "arr_recalls": arr_recalls.tolist(), 
                "preds": preds[:,1].tolist(), 'targets': targets.tolist(), "outputs" : outputs}
        
        def run_adaptation_and_evaluation(self, adaptset_dataloader, testset_dataloader, device):
            """
                Perform adaptation & testing jointly on adaptset and testset
            """
            loss = 0
            correct = 0
            tmpcnt = 0
            tmptotal = 0
            preds   = []
            targets = []
            outputs = []
            
            ## prepare model, optimizer for adaptation
            params = self.setup_adaptation()
            optimizer = optim.SGD(params,
                lr=self.lr,
                momentum=self.momentum,
                dampening=self.dampening,
                weight_decay=self.wd,
                nesterov=self.nesterov)
            
            for img, target in testset_dataloader:
                tmpcnt += 1
                tmptotal += len(target)
                img     = img.to(device)
                target  = target.to(device)
                for _ in range(self.test_steps):
                    output  = TentAdaptTest.forward_and_adapt(img, self.ResNet18, optimizer)

                if output.shape[1] == 1:
                    pred = torch.round(output)
                else:
                    pred = F.softmax(output, dim=1)
                    
                preds.append(pred.cpu().detach().numpy())
                targets.append(target.cpu().detach().numpy())
                outputs.append(output.cpu().detach().numpy())

            targets = np.concatenate(targets)
            preds   = np.concatenate(preds)
            outputs = np.concatenate(outputs)
            outputs = [outputs[i].tolist() for i in range(len(outputs))]
            preds_binary = preds.argmax(axis=1) 
            acc = (preds_binary == targets).mean()
            # Compute precision, recall, AUC, AP for binary classification
            # Plot the ROC
            fpr, tpr, _ = metrics.roc_curve(targets,   preds[:,1])
            # Plot the Recall-Precision Curve
            arr_precs, arr_recalls, threshold = metrics.precision_recall_curve(targets, preds[:,1])
            prec, rec, f1, sprt  = metrics.precision_recall_fscore_support(targets, preds_binary, average="binary")
            try:
                auc = metrics.roc_auc_score(targets, preds[:, 1])
            except:
                auc = None
            ap  = metrics.average_precision_score(targets, preds[:,1])
            return copy.deepcopy(self.state_dict()), loss, {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "ap": ap, 
                "tpr": tpr.tolist(), "fpr" : fpr.tolist(), "arr_precs": arr_precs.tolist(), "arr_recalls": arr_recalls.tolist(), 
                "preds": preds[:,1].tolist(), 'targets': targets.tolist(), "outputs" : outputs}
        
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
                
        def forward(self, x, perform_adapt=True):
            outputs = self.ResNet18(x)
            return outputs
        
    return TentAdaptTest