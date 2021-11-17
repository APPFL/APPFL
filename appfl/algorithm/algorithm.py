import copy
import torch

"""This implements a base class for server."""


class BaseServer:
    def __init__(self, model, num_clients, device):
        self.model = model
        self.num_clients = num_clients
        self.device = device

    # update global model
    def update(self):
        raise NotImplementedError

    def get_model(self):
        return copy.deepcopy(self.model)
 
    # NOTE: this is only for testing purpose.
    def validation(self):
        if self.loss_fn is None or self.dataloader is None:
            return 0.0, 0.0

        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        tmpcnt=0; tmptotal=0
        with torch.no_grad():
            for img, target in self.dataloader:
                tmpcnt+=1; tmptotal+=len(target)
                img = img.to(self.device)
                target = target.to(self.device)
                logits = self.model(img)                
                test_loss += self.loss_fn(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # FIXME: do we need to sent the model to cpu again?
        # self.model.to("cpu")
        
        test_loss = test_loss / tmpcnt
        accuracy = 100.0 * correct / tmptotal

        return test_loss, accuracy


"""This implements a base class for client."""


class BaseClient:
    def __init__(self, id, model, optimizer, optimizer_args, dataloader, device):
        self.id = id
        self.model = model
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.dataloader = dataloader
        self.device = device

    # update local model
    def update(self):
        raise NotImplementedError

    def get_model(self):
        return self.model.state_dict()
