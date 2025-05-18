import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def _one_hot(tensor, num):
    b = list(tensor.size())[0]
    onehot = torch.cuda.FloatTensor(b, num).fill_(0)
    ones = torch.cuda.FloatTensor(b, num).fill_(1)
    out = onehot.scatter_(1, torch.unsqueeze(tensor, 1), ones)
    return out


class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss


def cor(m):
    m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def reduction_loss(embedding, identity_matrix, size):
    loss = torch.mean(torch.abs(torch.triu(cor(embedding), diagonal=1)))
    loss = loss + 1 / torch.mean(
        torch.abs(embedding - torch.mean(embedding, dim=0).view(1, size).repeat(embedding.size()[0], 1)))
    loss = loss + torch.mean(torch.abs(embedding))
    return loss


class EncodingLoss(nn.Module):
    def __init__(self, dim=64, p =0.8, use_gpu = True):
        super(EncodingLoss, self).__init__()
        if use_gpu:
            self.identity_matrix = torch.tensor(np.identity(dim)).float().cuda()
        else:
            self.identity_matrix = torch.tensor(np.identity(dim)).float()
        self.p = p 
        self.dim = dim
        
    def forward(self, atac_embeddings, rna_embeddings):
        # rna
        rna_embedding_cat = rna_embeddings[0]
        rna_reduction_loss = reduction_loss(rna_embeddings[0], self.identity_matrix, self.dim)
        for i in range(1, len(rna_embeddings)):                
            rna_embedding_cat = torch.cat([rna_embedding_cat, rna_embeddings[i]], 0)            
            rna_reduction_loss += reduction_loss(rna_embeddings[i], self.identity_matrix, self.dim)                    
        rna_reduction_loss /= len(rna_embeddings)
        
        # atac
        atac_reduction_loss = reduction_loss(atac_embeddings[0], self.identity_matrix, self.dim)
        for i in range(1, len(atac_embeddings)):
            atac_reduction_loss +=  reduction_loss(atac_embeddings[i], self.identity_matrix, self.dim)                                
        atac_reduction_loss /= len(atac_embeddings)
        
        loss = rna_reduction_loss + atac_reduction_loss
        return loss

class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()

    def forward(self, rna_class_out, rna_class_label):
        rna_class_loss = F.cross_entropy(rna_class_out, rna_class_label.long())
        return rna_class_loss

