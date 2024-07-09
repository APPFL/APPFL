import torch.nn as nn

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)
    
    
if __name__ == '__main__':
    model = FC()

    # Calculate the total number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the number of parameters
    print(f'Total number of parameters: {count_parameters(model)}')

    # Calculate the model size in bytes and convert to Bytes
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        size_all_b = (param_size + buffer_size)
        return size_all_b

    # Print the model size
    print(f'Model size: {get_model_size(model):.2f} B')

