import torch
from torchvision.models import vit_l_16

def ViT():
    return vit_l_16(pretrained=False)


if __name__ == '__main__':
    model = ViT()

    # Calculate the total number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print the number of parameters
    total_params = count_parameters(model)
    print(f'Total number of parameters: {total_params}')

    # Calculate the model size in bytes and convert to MB
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.numel() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    # Print the model size
    model_size_mb = get_model_size(model)
    print(f'Model size: {model_size_mb:.2f} MB')

    # Example input to verify the model
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 input channels, 224x224 image
    output = model(input_tensor)
    print(f'Output shape: {output.shape}')
