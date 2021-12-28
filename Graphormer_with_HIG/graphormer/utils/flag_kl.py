# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import math
import torch.nn as nn

def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def consis_loss(logps, temp=0.5, lam=1.0):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)

    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss

def flag_bounded(model_forward, perturb_shape, y, optimizer, device, criterion, m=3, step_size=1e-3, mag=1e-3, mask=None):
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    with torch.no_grad():
        perturb = None
        out_ori = forward(perturb).view(-1)  
        if mask is not None:
            out_ori = out_ori[mask]  

    if mag > 0:
        perturb = torch.FloatTensor(*perturb_shape).uniform_(-1, 1).to(device)
        perturb = perturb * mag / math.sqrt(perturb_shape[-1])
    else:
        perturb = torch.FloatTensor(
            *perturb_shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb).view(-1)
    if mask is not None:
        out = out[mask]
    loss = criterion(out, y) +js_div(out_ori, out, get_softmax=True)
    loss /= m

    for _ in range(m-1):
        # loss.backward()
        model.manual_backward(loss)
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        if mag > 0:
            perturb_data_norm = torch.norm(perturb_data, dim=-1).detach()
            exceed_mask = (perturb_data_norm > mag).to(perturb_data)
            reweights = (mag / perturb_data_norm * exceed_mask +
                         (1-exceed_mask)).unsqueeze(-1)
            perturb_data = (perturb_data * reweights).detach()

        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb).view(-1)
        if mask is not None:
            out = out[mask]
        loss = criterion(out, y) +js_div(out_ori, out, get_softmax=True)
        loss /= m

    # loss.backward()
    model.manual_backward(loss)
    optimizer.step()

    return loss, out
