import torch
import torch.nn.functional as F
import numpy as np
import statistics as stats


def get_accuracy(logits, labels):
    out = F.log_softmax(logits, dim=1)
    _, predicted = torch.max(out, axis=1)
    correct = (predicted == labels.data)
    return correct


def get_small_count(data, variable):
    res = {}
    data = data.copy()
    dist = np.squeeze(data.groupby(variable).count().values)
    keys = np.sort(data[variable].unique())
    mean = stats.mean(dist)
    for i, value in enumerate(dist):
        if value < mean:
            res[i] = value
    return res, mean


def train_stats(epoch, num_epochs, batch_i, training_step, loss, acc):
    stats = 'Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Acc: %.4f' % (
                epoch, num_epochs, batch_i, training_step, loss, acc)
    return stats

def train_val_stats(epoch, num_epochs, batch_i, training_step, loss, acc, val_acc):
    stats = 'Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Acc: %.4f, Val Acc: %.4f' % (
                epoch, num_epochs, batch_i, training_step, loss, acc, val_acc)
    return stats
