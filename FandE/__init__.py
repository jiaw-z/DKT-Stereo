import torch
import random

def FandE_Ensemble(source, target, valid, clamp=False, threshold=3):
    valid_consistent = torch.sum((target - source)**2, dim=1).sqrt() < threshold
    valid_consistent = valid_consistent.unsqueeze(1) * valid
    source = source * valid
    target = target * valid
    offset = torch.sum((source - target)**2, dim=1).sqrt().unsqueeze(1)
    prob = random.random()
    assert prob >=0 and prob <=1, [prob]
    offset = prob * offset
    if clamp:
        offset = torch.clamp(offset, max=clamp)
    direction = torch.zeros_like(source)
    direction[source < target] = +1.
    direction[source > target] = -1.
    aug = direction * offset * valid_consistent
    AUG_source = source + aug
    AUG_source = AUG_source * valid
    return AUG_source
    

def FandE_Filter(source, target, valid, withprob=False, threshold=3):
    valid_consistent = torch.sum((target - source)**2, dim=1).sqrt() < threshold
    valid_consistent = valid_consistent.unsqueeze(1) * valid
    source = source * valid
    if withprob:
        num_valid_consistent = valid_consistent.flatten(1).sum(dim=-1, keepdim=True)
        num_valid = valid.flatten(1).sum(dim=-1, keepdim=True)
        prob_threshold = num_valid_consistent / num_valid
        prob = torch.rand(prob_threshold.shape).to(prob_threshold.device)
        valid_BinarySelect = (prob < prob_threshold).unsqueeze(-1).unsqueeze(-1)
        valid_BinarySelect = valid_BinarySelect * (1-valid_consistent) * valid
        Aug_valid = (valid_consistent + (1-valid_consistent) * valid_BinarySelect) * valid
    else:
        Aug_valid = valid_consistent
    AUG_source = source * Aug_valid
    return AUG_source, Aug_valid.squeeze(1)