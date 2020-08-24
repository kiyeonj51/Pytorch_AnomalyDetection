import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std= std

    def __call__(self, tensor):
        return tensor+torch.randn(tensor.size())*self.std+self.mean

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std{1}'.format(self.mean, self.std)


class AddSparseNoise:
    def __init__(self, prop=0.1):
        self.prop = prop

    def __call__(self, tensor):
        tensor[torch.rand(tensor.size())<self.prop]=1
        return tensor

    def __repr__(self):
        return self.__class__.__name__+'(prop={0}'.format(self.prop)