
from VAE import *
from data import load_data


def train(data, network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        n_samples = data.shape[0]
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = data[total_batch * i :
                            min(total_batch * i + batch_size, n_samples), :]

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
    return vae



def main():
    # network_architecture = dict(n_hidden_recog_1=4000, n_hidden_recog_2=1000, n_hidden_gener_1=1000, n_hidden_gener_2=4000, n_input=12562, n_z=500)
    network_architecture = dict(n_hidden_recog_1=4000, # 1st layer encoder neurons
                                n_hidden_recog_2=1000, # 2nd layer encoder neurons
                                n_hidden_gener_1=1000, # 1st layer decoder neurons
                                n_hidden_gener_2=4000, # 2nd layer decoder neurons
                                n_input=12562, n_z=500)
    data = load_data()[0:1000, :]
    vae = train(data, network_architecture, training_epochs=75)



