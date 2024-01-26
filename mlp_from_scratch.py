import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# set the seed
np.random.seed(123)

start = time.time()

# Training Data
x_train = pd.read_csv('data/training_set.csv', header=None).values
y_train = pd.read_csv('data/training_labels_bin.csv', header=None).values
x_val = pd.read_csv('data/validation_set.csv', header=None).values
y_val = pd.read_csv('data/validation_labels_bin.csv', header=None).values
N = len(x_train)

num_feats = x_train.shape[1]
n_out = y_train.shape[1]

# hyperparameters
eta = 0.1  # intial learning rate
gamma = 0.1  # multiplier for the learning rate
stepsize = 200  # epochs before changing learning rate
threshold = 0.08  # stopping criterion
test_interval = 10  # number of epoch before validating
max_epoch = 3000
hidden_layer_size = 64

# Define Architecture of NN
# Intialize your network weights and biases here
# We have two hidden layers and one output layer
w1 = np.random.randn(hidden_layer_size, num_feats)
b1 = np.random.randn(hidden_layer_size, 1)
w2 = np.random.randn(hidden_layer_size, hidden_layer_size)
b2 = np.random.randn(hidden_layer_size, 1)
w3 = np.random.randn(n_out, hidden_layer_size)
b3 = np.random.randn(n_out, 1)


# Define activation function here
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


train_loss = []
val_loss = []
best_val_loss = np.inf
no_improvement = 0

for epoch in range(0, max_epoch):

    order = np.random.permutation(N)  # shuffle data

    sse = 0
    for n in range(0, N):
        idx = order[n]

        # get a sample (batch size=1)
        x_in = np.array(x_train[idx]).reshape((num_feats, 1))
        y = np.array(y_train[idx]).reshape((n_out, 1))

        # do the forward pass here
        # hint: you need to save the output of each layer to calculate the gradients later
        z1 = np.matmul(w1, x_in) + b1
        h1 = sigmoid(z1)
        z2 = np.matmul(w2, h1) + b2
        h2 = sigmoid(z2)
        z3 = np.matmul(w3, h2) + b3
        y_hat = sigmoid(z3)

        # compute error and gradients here
        # hint: don't forget the chain rule
        squared_error = np.sum((y - y_hat) ** 2)
        # Output layer
        delta3 = (y_hat - y) * sigmoid_prime(z3)
        grad_w3 = np.matmul(delta3, h2.T)
        grad_b3 = delta3
        # Hidden layer 2
        delta2 = np.matmul(w3.T, delta3) * sigmoid_prime(z2)
        grad_w2 = np.matmul(delta2, h1.T)
        grad_b2 = delta2
        # Hidden layer 1
        delta1 = np.matmul(w2.T, delta2) * sigmoid_prime(z1)
        grad_w1 = np.matmul(delta1, x_in.T)
        grad_b1 = delta1

        # update weights and biases here
        # update weights and biases in output layer
        w3 = w3 - eta * grad_w3
        b3 = b3 - eta * grad_b3
        # update weights and biases in hidden layer 2
        w2 = w2 - eta * grad_w2
        b2 = b2 - eta * grad_b2
        # update weights and biases in hidden layer 1
        w1 = w1 - eta * grad_w1
        b1 = b1 - eta * grad_b1

        sse += squared_error

    train_mse = sse / len(x_train)

    if epoch % test_interval == 0:
        # test on validation set here
        val_sse = 0
        for i in range(len(x_val)):
            x_in = np.array(x_val[i]).reshape((num_feats, 1))
            y = np.array(y_val[i]).reshape((n_out, 1))

            # do the forward pass here
            # hint: you need to save the output of each layer to calculate the gradients later
            z1 = np.matmul(w1, x_in) + b1
            h1 = sigmoid(z1)
            z2 = np.matmul(w2, h1) + b2
            h2 = sigmoid(z2)
            z3 = np.matmul(w3, h2) + b3
            y_hat = sigmoid(z3)

            squared_error = np.sum((y - y_hat) ** 2)
            val_sse += squared_error
        val_mse = val_sse / len(x_val)
        print('Epoch: ' + str(epoch) + ' Train MSE: ' + str(train_mse) + ' Validation MSE: ' + str(val_mse))
        train_loss.append(train_mse)
        val_loss.append(val_mse)

        # if termination condition is satisfied, exit
        if val_mse < best_val_loss:
            best_val_loss = val_mse
        else:
            # Early stopping as validation loss is increasing
            print('Early stopping at epoch ' + str(epoch))
            break

    if epoch % stepsize == 0 and epoch != 0:
        eta = eta * gamma
        print('Changed learning rate to lr=' + str(eta))

end = time.time()
print('Time elapsed: ' + str(end - start))

# Plot the training and validation loss
plt.plot(np.arange(0, len(train_loss) * 10, 10), train_loss, label='Training Loss')
plt.plot(np.arange(0, len(val_loss) * 10, 10), val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig('training_validation_loss.pdf', dpi=300)
plt.show()
plt.close()
