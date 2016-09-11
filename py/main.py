
from VAE import *


def main():
    network_architecture = \
        dict(n_hidden_recog_1=500, # 1st layer encoder neurons
             n_hidden_recog_2=500, # 2nd layer encoder neurons
             n_hidden_gener_1=500, # 1st layer decoder neurons
             n_hidden_gener_2=500, # 2nd layer decoder neurons
             n_input=784, # MNIST data input (img shape: 28*28)
             n_z=20)  # dimensionality of latent space

    vae = train(network_architecture, training_epochs=75)


def main2():
    x_sample = mnist.test.next_batch(100)[0]
    x_reconstruct = vae.reconstruct(x_sample)

    plt.figure(figsize=(8, 12))
    for i in range(5):

        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()


def main3():
    network_architecture = \
                           dict(n_hidden_recog_1=500, # 1st layer encoder neurons
                                n_hidden_recog_2=500, # 2nd layer encoder neurons
                                n_hidden_gener_1=500, # 1st layer decoder neurons
                                n_hidden_gener_2=500, # 2nd layer decoder neurons
                                n_input=784, # MNIST data input (img shape: 28*28)
                                n_z=2)  # dimensionality of latent space
    
    vae_2d = train(network_architecture, training_epochs=75)
    
    x_sample, y_sample = mnist.test.next_batch(5000)
    z_mu = vae_2d.transform(x_sample)
    plt.figure(figsize=(8, 6)) 
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    plt.show()    



def main3():
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            x_mean = vae_2d.generate(z_mu)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()


