from gans import CGAN, load_mnist

z_dim = 64
img_size = (1, 28, 28)
lr = 1e-5
betas = (.5, .999)
n_epochs = 200
n_classes = 10

dataloader = load_mnist(path='./Data', batch_size=128)

g = CGAN()
g.setup(z_dim, n_classes, img_size, lr, betas)

#g.load_state('./Save/CGAN/gan-0002.h5')
g.train(dataloader, 20, save_step=20, path='./Save/CGAN')
