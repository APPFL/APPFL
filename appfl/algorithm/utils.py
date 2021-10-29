import torch

def validation(model, loss_fn, dataloader, device):
    if loss_fn is None or dataloader is None:
        return 0.0, 0.0

    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, target in dataloader:
            img = img.to(device)
            target = target.to(device)
            logits = model(img)
            test_loss += loss_fn(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # FIXME: do we need to sent the model to cpu again?
    # self.model.to("cpu")
    test_loss = test_loss / len(dataloader)
    accuracy = 100.0 * correct / len(dataloader.dataset)

    return test_loss, accuracy