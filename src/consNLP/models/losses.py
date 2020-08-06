import torch
import torch.nn as nn
import torch.nn.functional as F

def MSELoss(outputs, targets):
    return nn.MSELoss()(outputs, targets)

def MAELoss(outputs, targets):
    return nn.L1Loss()(outputs, targets)

def CELoss(outputs, targets):
    return nn.CrossEntropyLoss()(outputs.view(-1,outputs.shape[-1]), targets.view(-1,1).view(-1).type_as(targets))

class masked_CELoss(nn.Module):
    def __init__(self):
        super(masked_CELoss, self).__init__()
        self.lfn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, masks):
        active_loss = masks.view(-1) == 1
        active_logits = outputs.view(-1, outputs.shape[-1])
        active_labels = torch.where(
            active_loss,
            targets.view(-1),
            torch.tensor(self.lfn.ignore_index).type_as(targets)
        )
        loss = self.lfn(active_logits, active_labels.type_as(targets))

        return loss

def BCELoss(outputs, targets):
    return nn.BCELoss()(outputs, targets.view(-1, 1))

def BCEWithLogitsLoss(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def KLDivLoss(outputs, targets):
    if len(targets.size) == 1:
        return nn.KLDivLoss()(outputs, targets.view(-1, 1))
    else:
        return nn.KLDivLoss()(outputs, targets)

def QACELoss(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)

    return total_loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)       
        
        #flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (outputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(outputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)       
        
        #flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (outputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(outputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(outputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)       
        
        #flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (outputs * targets).sum()
        total = (outputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, outputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)       
        
        #flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(outputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class OHEMLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(OHEMLoss, self).__init__()

    def forward(self, outputs, targets, rate):
        batch_size = targets.size(0) 
        ohem_cls_loss = F.cross_entropy(outputs, targets, reduction='none', ignore_index=-1)

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num

        return cls_loss

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

#=====
#Multi-class Lovasz loss
#=====

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, outputs, targets):
        outputs = F.sigmoid(outputs)    
        Lovasz = lovasz_hinge(outputs, targets, per_image=False)                       
        return Lovasz

LOSSES = {'bce': BCELoss,
         'bcelogit': BCEWithLogitsLoss,
         'ce': CELoss,
         'kld': KLDivLoss,
         'dice': DiceLoss,
         'dicewithbce': DiceBCELoss,
         'mse': MSELoss,
         'mae': MAELoss,
         'masked_ce': masked_CELoss(),
         'qa_ce': QACELoss}

def get_loss(loss_name):
    """Get a loss from string.
    Parameters
    ----------
    loss_name : str | callable
        loss function method as string. If callable it is returned as is.
    Returns
    -------
    loss function : callable
        The loss function.
    """
    if isinstance(loss_name, str):
        try:
            loss_fn = LOSSES[loss_name]
        except KeyError:
            raise ValueError('{} is not a valid loss function value. '
                             'Use sorted({}) '
                             'to get valid options.'.format(loss_name,LOSSES.keys()))
    else:
        loss_fn = loss_name
        
    return loss_fn