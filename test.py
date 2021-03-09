from gans import GAN, load_mnist
import torch

def test_train_ground():
    z_dim = 64
    img_size = (1, 28, 28)
    lr = 1e-5
    betas = (.5, .999)
    n_epochs = 200
    n_classes = 10

    dataloader = load_mnist(path='./data', batch_size=128)

    g = CGAN()
    g.setup(z_dim, n_classes, img_size, lr, betas)
    g.train(dataloader, n_epochs, path='./save/cgan')

def test_train_load():
    dataloader = load_mnist(path='./data', batch_size=128)
    n_epochs = 200

    g = CGAN()
    g.load_state('./save/cgan/cgan-1000.h5')
    g.train(dataloader, n_epochs, path='./save/cgan')

def test_new_data():
    g = GAN()
    g.load_state('./save/gan/gan-1000.h5')
    
    z = g.noise(200)
    
    x = g.generator(z)

    g.show_images(x, nrow=20, figsize=(20, 10), path='img/gan.png')

def test_latent_perturbing():
    pass

test_new_data()