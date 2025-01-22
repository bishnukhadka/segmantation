import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import numpy as np


class SegmentationLosses(object):
    def __init__(self, weight=None, 
    size_average=True, 
    batch_average=True, 
    ignore_index=255, 
    cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal' or 'jacardian']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'jacardian':
            return self.Jacardian_loss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        # if isinstance(logit, OrderedDict):
        #     logit = logit['out']
        
        n, c, h, w = logit.size()
            
        criterion = nn.CrossEntropyLoss(weight=self.weight, 
                                        ignore_index=self.ignore_index,
                                        # size_average=self.size_average
                                        )
        if self.cuda:
            criterion = criterion.cuda()
    
        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, 
                                        ignore_index=self.ignore_index,
                                        # size_average=self.size_average
                                        )
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss
    
    def Jacardian_loss(self, logit, target, smooth=100):
        probs = torch.softmax(logit, dim=1)  # Softmax along the class dimension
        # Take the class with the highest probability
        y_pred = torch.argmax(probs, dim=1)  # Get the index of the maximum probability
        c_mat = confusion_matrix(y_pred=y_pred.numpy(), y_true=target.numpy())
        miou_scores = []
        for i in range(self.num_class-1):
            i=i+1
            tp = c_mat[i, i]
            fp = np.sum(c_mat[:, i]) - tp
            fn = np.sum(c_mat[i, :]) - tp
            miou = (tp) / (tp + fp + fn)
            miou_scores.append(miou)
        miou=np.nanmean(miou_scores)
        return (1-miou)*smooth


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())



