import keras
import matplotlib.pyplot as plt
import math
import numpy as np


def plot(update_steps, accuracies, losses, costs, val_accuracies, val_losses, val_costs):
    """Plot accuracy, loss, cost."""
    _, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].plot(update_steps, accuracies, linestyle='-', color='blue', label='Train Accuracy')
    axs[0].plot(update_steps, val_accuracies, linestyle='-', color='red', label='Validation Accuracy')
    axs[0].set_ylim(0, None)
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Update Steps')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(update_steps, losses, linestyle='-', color='blue', label='Train Loss')
    axs[1].plot(update_steps, val_losses, linestyle='-', color='red', label='Validation Loss')
    axs[1].set_ylim(0, None)
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Update Steps')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    axs[2].plot(update_steps, costs, linestyle='-', color='blue', label='Train Cost')
    axs[2].plot(update_steps, val_costs, linestyle='-', color='red', label='Validation Cost')
    axs[2].set_ylim(0, None)
    axs[2].set_title('Cost')
    axs[2].set_xlabel('Update Steps')
    axs[2].set_ylabel('Cost')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


class Cifar10: 
    def __init__(self):
        self.labels = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }

        # X_train: non-processed training images, shape(50000, 32, 32, 3)
        # y_train: training labels with values 0-9, shape(50000, 1)
        # X_test: non-processed test images, shape(10000, 32, 32, 3)
        # y_test: test labels with values 0-9, shape(10000, 1)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = \
            keras.datasets.cifar10.load_data()
        
        # X_train_std: processed training images, shape(3072, 50000)
        # Y_train: onehot encoded training labels, shape(10, 50000)
        # X_test_std: processed test images, shape(3072, 10000)
        # Y_test: onehot encoded test labels, shape(10, 10000)
        (self.X_train_std, self.Y_train), (self.X_test_std, self.Y_test) = \
            self.preprocess()
        
        
    def preprocess(self):
        """Preprocesses the CIFAR-10 dataset."""
        # Convert type: uint8 -> float64
        X_train_std = self.X_train.astype('float64')
        X_test_std = self.X_test.astype('float64')

        # Normalize values: 0-255 -> 0-1
        X_train_std = X_train_std / 255.0
        X_test_std = X_test_std / 255.0

        # Flatten images: 50000x32x32x3 -> 50000x3072
        X_train_std = X_train_std.reshape(X_train_std.shape[0], -1)
        X_test_std = X_test_std.reshape(X_test_std.shape[0], -1)

        # One-hot encode labels: 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Y_train = np.zeros((50000, 10))
        Y_train[np.arange(50000), self.y_train.flatten()] = 1
        Y_test = np.zeros((10000, 10))
        Y_test[np.arange(10000), self.y_test.flatten()] = 1

        # Standardize data to have a mean of 0 and standard deviation of 1
        X_train_means = np.mean(X_train_std, axis=0).reshape(1, -1)
        X_train_standard_devs = np.std(X_train_std, axis=0).reshape(1, -1)
        X_train_std = (X_train_std - X_train_means) / X_train_standard_devs
        X_test_means = np.mean(X_test_std, axis=0).reshape(1, -1)
        X_test_standard_devs = np.std(X_test_std, axis=0).reshape(1, -1)
        X_test_std = (X_test_std - X_test_means) / X_test_standard_devs

        # Transpose data to match assignment description and lectures
        X_train_std = X_train_std.T
        Y_train = Y_train.T
        X_test_std = X_test_std.T
        Y_test = Y_test.T

        return (X_train_std, Y_train), (X_test_std, Y_test)
    

    def get_data(self):
        """Returns the unprocessed training and testing data."""
        return self.X_train, self.y_train, self.X_test, self.y_test
    

    def get_preprocessed_data(self):
        """Returns the preprocessed training and testing data."""
        return self.X_train_std, self.Y_train, self.X_test_std, self.Y_test


class NeuralNetwork:
    def __init__(self):
        # (m, 3072) x (3072, 1) -> (m, 1)
        self.W1 = self.generate_weights((50, 3072), 1.0/math.sqrt(3072))
        # (m, 1)
        self.b1 = self.generate_biases((50, 1))
        # (K, m) x (m, 1) -> (K, 1)
        self.W2 = self.generate_weights((10, 50), 1.0/math.sqrt(50))
        # (K, 1)
        self.b2 = self.generate_biases((10, 1))


    @staticmethod
    def generate_weights(shape, standard_dev):
        """Generates weights with Gaussian random values."""
        return np.random.normal(loc=0.0, scale=standard_dev, size=shape)


    @staticmethod
    def generate_biases(shape):
        """Generates biases with zero values."""
        return np.zeros(shape)
    

    @staticmethod
    def generate_cyclical_learning_rates(eta_min, eta_max, ns, num_cycles):
        """Generate a list of cyclical learning rates"""
        t = 0
        learning_rates = []
        for l in range(num_cycles): 
            for _ in range(ns):
                eta_t = eta_min + ((t-2*l*ns)/ns) * (eta_max-eta_min)
                learning_rates.append(eta_t)
                t += 1
            for _ in range(ns):
                eta_t = eta_max - ((t-(2*l+1)*ns)/ns) * (eta_max-eta_min)
                learning_rates.append(eta_t)
                t += 1
        return learning_rates


    @staticmethod
    def relu(S):
        """Compute the ReLU function."""
        return np.maximum(0, S)


    @staticmethod
    def softmax(S):
        """Compute the softmax activation function."""
        exp_s = np.exp(S)
        return exp_s / np.sum(exp_s, axis=0, keepdims=True)
    

    @staticmethod
    def cross_entropy_loss(Y, P):
        """Compute the average cross-entropy loss for a batch."""
        return np.sum(-np.sum(Y * np.log(P), axis=0)) / Y.shape[1]
    

    def l2_regularization(self):
        """Compute L2 regularization for the network."""
        return np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)
    

    def evaluate_classifier(self, X):
        """Compute the forward pass."""
        H = self.relu(self.W1 @ X + self.b1)
        P = self.softmax(self.W2 @ H + self.b2)
        return H, P
    

    def compute_cost(self, X, Y, λ):
        """Compute the cost function."""
        _, P = self.evaluate_classifier(X)
        return self.cross_entropy_loss(Y, P) + λ*self.l2_regularization()
    

    def compute_accuracy(self, X, y):
        """Compute the accuracy of the model."""
        _, P = self.evaluate_classifier(X)
        return np.sum(np.argmax(P, axis=0) == y.flatten()) / P.shape[1]
    

    def compute_gradients(self, X, Y, λ):
        """Compute the gradients of the loss function."""
        batch_size = X.shape[1]
        H, P = self.evaluate_classifier(X)

        G = -(Y - P)

        grad_W2 = (1 / batch_size) * G @ H.T + (2*λ) * self.W2
        grad_b2 = np.mean(G, axis=1).reshape(-1, 1)

        G = self.W2.T @ G 
        G = G * np.where(H > 0, 1, 0)

        grad_W1 = (1 / batch_size) * G @ X.T + (2*λ) * self.W1
        grad_b1 = np.mean(G, axis=1).reshape(-1, 1)

        return grad_W1, grad_b1, grad_W2, grad_b2 
    

    def update_parameters(self, grad_W1, grad_b1, grad_W2, grad_b2, learning_rate):
        """Update the models parameters based on learning rate and gradients."""
        self.W1 = self.W1 - learning_rate * grad_W1
        self.b1 = self.b1 - learning_rate * grad_b1
        self.W2 = self.W2 - learning_rate * grad_W2
        self.b2 = self.b2 - learning_rate * grad_b2


    def train(self, X, Y, y, X_val, Y_val, y_val, learning_rates, batch_size, λ, n, ns):
        update_steps = []
        accuracies = []
        costs = []
        losses = []
        val_accuracies = []
        val_costs = []
        val_losses = []

        t = 0
        for learning_rate in learning_rates:
            if t % 100 == 0:
                update_steps.append(t)

                acc = self.compute_accuracy(X, y)
                cost = self.compute_cost(X, Y, λ)
                _, P = self.evaluate_classifier(X)
                loss = self.cross_entropy_loss(Y, P)

                accuracies.append(acc)
                costs.append(cost)
                losses.append(loss)

                val_acc = self.compute_accuracy(X_val, y_val)
                val_cost = self.compute_cost(X_val, Y_val, λ)
                _, P_val = self.evaluate_classifier(X_val)
                val_loss = self.cross_entropy_loss(Y_val, P_val)

                val_accuracies.append(val_acc)
                val_costs.append(val_cost)
                val_losses.append(val_loss)

                print()
                print(f't = {t}')
                print(f' - training data: acc {acc:.4f}, cost {cost:.4f} loss {loss:.4f}')
                print(f' - val data:     acc {val_acc:.4f}, cost {val_cost:.4f} loss {val_loss:.4f}')

            start = ((t)*batch_size) % n
            end = min(start+batch_size, X.shape[1])

            W1_grad, b1_grad, W2_grad, b2_grad = self.compute_gradients(X[:, start:end], Y[:, start:end], λ)
            self.update_parameters(W1_grad, b1_grad, W2_grad, b2_grad, learning_rate)

            t += 1

        update_steps.append(t)

        acc = self.compute_accuracy(X, y)
        cost = self.compute_cost(X, Y, λ)
        _, P = self.evaluate_classifier(X)
        loss = self.cross_entropy_loss(Y, P)

        accuracies.append(acc)
        costs.append(cost)
        losses.append(loss)

        val_acc = self.compute_accuracy(X_val, y_val)
        val_cost = self.compute_cost(X_val, Y_val, λ)
        _, P_val = self.evaluate_classifier(X_val)
        val_loss = self.cross_entropy_loss(Y_val, P_val)

        val_accuracies.append(val_acc)
        val_costs.append(val_cost)
        val_losses.append(val_loss)

        with open("out.txt", "a") as file:
            file.write("\n")
            file.write(f't = {t}, lambda = {λ}, ns = {ns}\n')
            file.write(f' - training data: acc {acc:.4f}, cost {cost:.4f} loss {loss:.4f}\n')
            file.write(f' - val data:     acc: {val_acc:.4f}, cost: {val_cost:.4f}, loss: {val_loss:.4f}\n')

        return update_steps, accuracies, losses, costs, val_accuracies, val_losses, val_costs


def main():
    # Get data.
    data = Cifar10()
    labels = data.labels
    X_train, y_train, X_test, y_test = data.get_data()
    X_train_std, Y_train, X_test_std, Y_test = data.get_preprocessed_data()
    
    # Set hyperparameters.
    eta_min = 1e-6
    eta_max = 1e-1
    ns = 2000
    num_cycles = 4
    learning_rates = NeuralNetwork.generate_cyclical_learning_rates(eta_min, eta_max, ns, num_cycles)
    batch_size = 100
    λ = 0.003540328811048995
    n = 49000

    X_use = X_train_std[:, :n]
    Y_use = Y_train[:, :n]
    y_use = y_train[:n]
    X_val = X_train_std[:, 49000:50000]
    Y_val = Y_train[:, 49000:50000]
    y_val = y_train[49000:50000]
    
    nn = NeuralNetwork()
    update_steps, accuracies, losses, costs, val_accuracies, val_losses, val_costs = \
        nn.train(X_use, Y_use, y_use, X_val, Y_val, y_val, learning_rates, batch_size, λ, n, ns)
        
    acc = nn.compute_accuracy(X_test_std, y_test)
    cost = nn.compute_cost(X_test_std, Y_test, λ)
    _, P = nn.evaluate_classifier(X_test_std)
    loss = nn.cross_entropy_loss(Y_test, P)

    print(f'test: {acc} {cost} {loss}')

    plot(update_steps, accuracies, losses, costs, val_accuracies, val_losses, val_costs)
    

if __name__ == "__main__":
    main()




        