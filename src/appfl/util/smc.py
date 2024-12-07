import crypten
crypten.init()

# # Define the CNN
# class TempCNN(nn.Module):
#     def __init__(self):
#         super(TempCNN, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Adjust based on input image size
#         self.fc2 = nn.Linear(128, 10)  # For 10 output classes
        
#         # Pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view( x.size(0), -1)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

def model_encryption(model):
    s_dict = model.state_dict()
    tensor_dict = {}
    for n, tensor in s_dict.items():
        e_t = crypten.cryptensor(tensor)
        tensor_dict[n]=e_t
    return tensor_dict
    
def model_decryption(tensor_dict, model):
    for k in tensor_dict:
        tensor_dict[k] = tensor_dict[k].get_plain_text()
    model.load_state_dict(tensor_dict)
    return model

def smc_encrypted_model_aggregation(list_of_e_models, weights=None):
    avg_model = {}
    for i, model in enumerate(list_of_e_models):
        for k in model:
            if avg_model.get(k) is None:
                avg_model[k] = model[k] * weights[i]
            else:
                avg_model[k] +=  model[k] * weights[i]
    for k in avg_model:
        avg_model[k]/=sum(weights)
    return avg_model