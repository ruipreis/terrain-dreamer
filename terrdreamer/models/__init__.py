import torch.nn as nn


def replace_batchnorm2d_with_instancenorm(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            instancenorm = nn.InstanceNorm2d(module.num_features, affine=True)
            instancenorm.weight = module.weight
            instancenorm.bias = module.bias
            setattr(model, name, instancenorm)
        else:
            replace_batchnorm2d_with_instancenorm(module)
