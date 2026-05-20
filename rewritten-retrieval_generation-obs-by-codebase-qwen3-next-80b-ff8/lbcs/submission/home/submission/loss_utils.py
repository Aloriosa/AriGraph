import torch
import torch.nn.functional as F
import numpy as np

def cross_entropy(K, alpha, y, weights, lmbda):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
    if lmbda > 0:
        loss_value += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss_value

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def compute_gradient_norms(model, data_loader, device):
    """Compute gradient norms for each sample in the dataset"""
    model.eval()
    gradient_norms = []
    
    for batch_idx, (data, target, indices) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        output = model(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass to compute gradients
        model.zero_grad()
        loss.backward()
        
        # Compute gradient norms for each sample
        for param in model.parameters():
            if param.grad is not None:
                # Compute norm of gradients for each sample
                grad_norm = torch.norm(param.grad.view(param.grad.size(0), -1), dim=1)
                gradient_norms.append(grad_norm.cpu().numpy())
    
    # Combine all gradient norms
    if len(gradient_norms) > 0:
        return np.concatenate(gradient_norms)
    return np.array([])