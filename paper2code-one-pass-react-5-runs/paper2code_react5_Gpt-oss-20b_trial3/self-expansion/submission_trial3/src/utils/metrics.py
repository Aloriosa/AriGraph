import torch

def accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def forgetting(all_task_accuracies):
    """
    Compute average forgetting over all seen tasks.
    `all_task_accuracies` is a list of lists of accuracies
    after each task. For example, acc[i][j] is accuracy on task i
    after training on task j (j>=i).
    """
    num_tasks = len(all_task_accuracies)
    forgetting = 0.0
    for i in range(num_tasks - 1):
        max_acc = max(all_task_accuracies[i])
        final_acc = all_task_accuracies[i][-1]
        forgetting += max_acc - final_acc
    return forgetting / (num_tasks - 1)