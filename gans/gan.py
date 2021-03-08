import torch
import torch.nn as nn
from tqdm.auto import tqdm 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

__all__ = ["Discriminator", "Generator", "GAN"]

criterion  = nn.BCEWithLogitsLoss()
hidden_dim = 128

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 4 * hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.BatchNorm1d(4 * hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(4 * hidden_dim, 8 * hidden_dim),
            nn.BatchNorm1d(8 * hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(8 * hidden_dim, out_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.model(z)

class GAN:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # img_size of the form (1, w, h)    e.g. for MNIST it's (1, 28, 28)
    def setup(self, z_dim, img_size, lr):
        self.z_dim = z_dim
        self.img_size = img_size
        assert len(img_size) == 3, 'size sould be of format : (channel, width, heigt)'

        x_dim = img_size[1] * img_size[2]
        self.generator = Generator(z_dim, x_dim).to(self.device)
        self.discriminator = Discriminator(x_dim).to(self.device)

        self.g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        self.d_loss_history = []
        self.g_loss_history = []
        self.z = self.noise(100)
        self.start_epoch = 0

    def load_state(self, path):
        state = torch.load(path, map_location=self.device)
        self.z_dim = state['z_dim']
        self.img_size = state['img_size']

        self.generator = state['gen']
        self.discriminator = state['disc']
        self.g_opt = state['g_opt']
        self.d_opt = state['d_opt']

        self.d_loss_history = state['d_loss_history'].tolist()
        self.g_loss_history = state['g_loss_history'].tolist()
        self.z = state['z']
        self.start_epoch = state['start_epoch'] 

    def noise(self, n):
        return torch.randn(n, self.z_dim, device=self.device)

    def show_images(self, images, path='.'):
        img_unflat = images.detach().cpu().view(-1, *self.img_size)
        img_grid = make_grid(img_unflat, nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_grid.permute(1, 2, 0).squeeze())
        plt.savefig(path)
        plt.close(None)

    def get_discriminator_loss(self, real, batch_size):
        noise = self.noise(batch_size)
        fake_image_gen = self.generator(noise)
        fake_image_pred = self.discriminator(fake_image_gen.detach())
        fake_image_loss = criterion(fake_image_pred, torch.zeros_like(fake_image_pred))
    
        real_image_pred = self.discriminator(real)
        real_image_loss = criterion(real_image_pred, torch.ones_like(real_image_pred))
        disc_loss = (fake_image_loss + real_image_loss) / 2
        return disc_loss

    def get_generator_loss(self, batch_size):
        noise = self.noise(batch_size)
        fake_image_gen = self.generator(noise)
        fake_image_pred = self.discriminator(fake_image_gen)

        gen_loss = criterion(fake_image_pred, torch.ones_like(fake_image_pred))
        return gen_loss

    def train(self, dataloader, n_epochs, display_step=1, save_step=50, path='.'):
        for epoch in range(self.start_epoch, n_epochs + 1):
            for real, _ in tqdm(dataloader):
                batch_size = len(real)
                real = real.view(batch_size, -1).to(self.device) # flatten

                """ Update discriminator """
                self.d_opt.zero_grad()
                disc_loss = self.get_discriminator_loss(real, batch_size)
                disc_loss.backward()
                self.d_opt.step()
                self.d_loss_history += [disc_loss.item()]

                """ Update generator """
                self.g_opt.zero_grad()
                gen_loss = self.get_generator_loss(batch_size)
                gen_loss.backward()
                self.g_opt.step()
                self.g_loss_history += [gen_loss.item()]

            ### Some visuals ###
            if epoch % display_step == 0:
                print(f"Epoch {epoch}: G_loss = {self.g_loss_history[-1]}, D_loss = {self.d_loss_history[-1]}")
                generated = self.generator(self.z)
                self.show_images(generated, path=path+'/sample-%04d.png'%epoch)
                # loss functions
                step_bins = 20
                n_example = (len(self.d_loss_history) // step_bins) * step_bins
                plt.clf()
                plt.figure(figsize=(10, 5))
                plt.plot(
                    range(n_example // step_bins),
                    torch.Tensor(self.g_loss_history[:n_example]).view(-1, step_bins).mean(1),
                    label="Generator loss"
                )
                plt.plot(
                    range(n_example // step_bins),
                    torch.Tensor(self.d_loss_history[:n_example]).view(-1, step_bins).mean(1),
                    label="Discriminator loss"
                )
                plt.legend()
                plt.savefig(path+'/loss-%04d.png'%epoch)
                plt.close(None)
                
            ### Model saving ###
            if epoch % save_step == 0:
                state = {
                    'z_dim': self.z_dim,
                    'img_size': self.img_size,
                    'gen': self.generator,
                    'disc': self.discriminator,
                    'd_opt': self.d_opt,
                    'g_opt': self.g_opt,
                    'd_loss_history': torch.Tensor(self.d_loss_history),
                    'g_loss_history': torch.Tensor(self.g_loss_history),
                    'z': self.z,
                    'start_epoch': epoch + 1,
                }
                torch.save(state, path+'/gan-%04d.h5'%epoch)