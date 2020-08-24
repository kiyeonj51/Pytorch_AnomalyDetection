import argparse
import os
import numpy as np

from torchvision.utils import save_image
from torch.autograd import Variable
import torch

from anogan.model import Generator, Discriminator
from anogan.util import weights_init_normal
from anogan.data_loader import mnist, energy


os.makedirs("images", exist_ok=True)
os.makedirs("saved_model", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--dataset", type=str, default="energy", help="data path")
parser.add_argument("--data_path", type=str, default="../data/energy/energydata_complete.csv", help="data path")
parser.add_argument("--window_size", type=int, default=28, help="window size of time series data")
parser.add_argument("--skip_size", type=int, default=7, help="skip size of time series data")
parser.add_argument("--train", action='store_true', help="is this train?")
parser.add_argument("--download", action='store_true', help="is this downloading data?")
parser.add_argument("--noise_level", type=float, default=0., help="interval between image sampling")
parser.add_argument("--add_noise", action='store_true', help="is adding noise?")

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

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader

training_generator = eval(opt.dataset)(opt)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(training_generator):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs)[0], valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs)[0], valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach())[0], fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(training_generator), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(training_generator) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], f"images/{opt.dataset}_{batches_done}.png", nrow=5, normalize=True)
            torch.save(generator.state_dict(), f'saved_model/{opt.dataset}_generator.pkl')
            torch.save(discriminator.state_dict(), f'saved_model/{opt.dataset}_discriminator.pkl')
