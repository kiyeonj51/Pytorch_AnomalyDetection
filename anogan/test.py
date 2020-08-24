import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

import torch

from anogan.model import Generator, Discriminator
from anogan.data_loader import mnist, energy

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.1, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--noise_level", type=float, default=0.1, help="interval between image sampling")
parser.add_argument("--add_noise", action='store_true', help="is adding noise?")
parser.add_argument("--lamb", type=float, default=0.1, help="weight of anomaly score")
parser.add_argument("--thres", type=float, default=0.2, help="threshold for anomaly detection")
parser.add_argument("--train", action='store_true', help="is this train?")
parser.add_argument("--download", action='store_true', help="is this downloading data?")

parser.add_argument("--dataset", type=str, default="energy", help="data path")
parser.add_argument("--data_path", type=str, default="../data/energy/energydata_complete.csv", help="data path")
parser.add_argument("--window_size", type=int, default=28, help="window size of time series data")
parser.add_argument("--skip_size", type=int, default=7, help="skip size of time series data")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Load weights
if os.listdir('saved_model'):
    generator.load_state_dict(torch.load(f'saved_model/{opt.dataset}_generator.pkl'))
    discriminator.load_state_dict(torch.load(f'saved_model/{opt.dataset}_discriminator.pkl'))

# Configure data loader
training_generator = eval(opt.dataset)(opt)
test_iter = iter(training_generator)
test_data_mnist = next(test_iter)[0]

# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
z = Variable(Tensor(np.random.normal(0,0.1,(opt.batch_size, opt.latent_dim))), requires_grad=True)
target = Variable(test_data_mnist.type(Tensor), requires_grad=False)
optimizer_A = torch.optim.Adam([z], lr=opt.lr)

# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def anomaly_score(x, G_z, lamb=0.1):
    residual_loss = torch.norm(x-G_z, p=1)
    discrimination_loss = torch.norm(discriminator(x)[1] - discriminator(G_z)[1], p=1)
    total_loss = (1-lamb) * residual_loss + lamb * discrimination_loss
    return total_loss


generator.eval()
discriminator.eval()

for it in range(opt.max_iter):
    gen_imgs = generator(z)
    a_loss = anomaly_score(target, gen_imgs, opt.lamb)
    optimizer_A.zero_grad()
    a_loss.backward()
    optimizer_A.step()
    print(f"[Iteration {it:4d}/{opt.max_iter}][A loss: {a_loss.item():6.5f}]")

def img_scaling(img):
    img = (img + 1.) * 255. / 2.
    return img.type(torch.uint8)

for idx in range(opt.batch_size):
    fig, axs = plt.subplots(1,4, figsize=(12,3))
    real = test_data_mnist[idx,0,:,:]
    gen = gen_imgs.cpu().data[idx,0,:,:]
    diff = (real - gen) / 2.
    anom = real.clone()
    anom[torch.abs(diff) > opt.thres] = -1
    anom_rgb = torch.zeros(opt.img_size, opt.img_size, 3)
    anom_rgb[:, :, 0] = img_scaling(anom)
    anom_rgb[:, :, 1] = img_scaling(real)
    anom_rgb[:, :, 2] = img_scaling(real)
    anom_rgb = anom_rgb.type(torch.uint8)

    imgs = [real.numpy(), gen.numpy(), diff.numpy(), anom_rgb.numpy()]

    for ax, interp, img in zip(axs, ['test data(+noise)', 'generated data', 'difference', 'anomaly'],imgs):
        ax.imshow(img, cmap="gray")
        ax.set_title(interp.capitalize())
    plt.savefig(f"images/restored_{opt.dataset}_{idx:2d}.png")
    plt.show()
