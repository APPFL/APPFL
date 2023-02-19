import torch
import random
import numpy as np

def add_noise(args, model, mode= "laplace"):
    sensitivity = cal_sensitivity(args.optim_args.lr, args.clip_value)
    if mode == 'laplace':
        with torch.no_grad():
            for k, v in model.named_parameters():
                if args.epsilon != False:
                    noise = Laplace(epsilon=args.epsilon, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(v.device)                               
                    v += noise
                
    # elif self.dp_mechanism == 'Gaussian':
    #     with torch.no_grad():
    #         for k, v in net.named_parameters():
    #             noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity, size=v.shape)
    #             noise = torch.from_numpy(noise).to(self.args.device)
    #             v += noise

def clip_gradients(args, model):
    if args.dp == 'laplace':
        # Laplace use 1 norm
        for k, v in model.named_parameters():
            v.grad /= max(1, v.grad.norm(1) / args.clip_value)
    # elif self.dp_mechanism == 'Gaussian':
    #     # Gaussian use 2 norm
    #     for k, v in net.named_parameters():
    #         v.grad /= max(1, v.grad.norm(2) / self.dp_clip)

def cal_sensitivity(lr, clip):
#     return 2 * lr * clip / dataset_size
    return 2 * lr * clip

def Laplace(epsilon, sensitivity, size):
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)

def set_seed(seed=233):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def training(model, train_loader, train_loss, correct, total, device, optimizer, criterion):

    for i, (images, labels) in enumerate(train_loader):         

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()    
 
        optimizer.step() 

        #compute        
        train_loss += loss.data.item()  
 
        #Get prediction
        _, predicted = torch.max(outputs.data,1)
        
        #total number of labels
        total += labels.size(0)
        
        #Total correct predictions
        correct += (predicted ==labels).sum()

    return model, train_loss, correct, total 



def lbfgs_training(model, train_loader, train_loss, correct, total, device, optimizer, criterion):
    for i, (images, labels) in enumerate(train_loader):         
        images = images.to(device)
        labels = labels.to(device)
  
        def closure():
            #Clear gradients, not be accumulated
            optimizer.zero_grad()
    
            #Forward pass to get output
            outputs = model(images)

    
            #Calculate Loss: softmax + cross entropy loss
            loss = criterion(outputs, labels)
    
            #Get gradients 
            loss.backward()
   
            return loss
 
        #update parameters
        loss = optimizer.step(closure)

        #compute
        outputs = model(images)      
        loss = criterion(outputs, labels)  
        train_loss += loss.data.item() 

  
        #Get prediction
        _, predicted = torch.max(outputs.data,1)
        
        #total number of labels
        total += labels.size(0)
        
        #Total correct predictions
        correct += (predicted ==labels).sum()
      
    return model, train_loss, correct, total

 

def validation(model,criterion,test_loader,device):
    model.eval()
    loss=0
    correct = 0 
    total = 0
    #Iterate through test data set
    for images, labels in test_loader:
        #Load images to a Torch Variable
        # images = Variable(images.view(-1, 28*28))
        
        images = images.to(device)
        labels = labels.to(device)            
        
        #Forward pass only to get output
        if model.name == "CNN_rgap" :
            outputs,shape = model(images)
            del images, shape
        else: 
            outputs = model(images)
            del images 
        loss += criterion(outputs, labels).detach().to("cpu")
        
        #Get prediction
        _, predicted = torch.max(outputs.data,1)
        
        del outputs
        #total number of labels
        total += labels.size(0)
        
        #Total correct predictions
        correct += (predicted ==labels).sum().to("cpu")
        del labels
    accuracy = 100*correct /total
    test_loss = loss / len(test_loader)
    del loss

    torch.cuda.empty_cache()
    
    return test_loss, accuracy


class BNStatisticsHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook to compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        # r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
        #     module.running_mean.data - mean, 2)
        mean_var = [mean, var]

        self.mean_var = mean_var
        # must have no output

    def close(self):
        self.hook.remove()


 