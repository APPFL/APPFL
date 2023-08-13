def get_loss():
    import torch
    import torch.nn as nn
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean(0)

    class SoftmaxEntropyCrossEntropyLoss(nn.CrossEntropyLoss):
        def forward(self, input, target):
            return super().forward(input, target) + softmax_entropy(input)
    return SoftmaxEntropyCrossEntropyLoss