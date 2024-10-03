import keras
import matplotlib.pyplot as plt
import math
import numpy as np


def plot(update_steps, train_accuracies, train_losses, train_costs, val_accuracies, val_losses, val_costs):
    _, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].plot(update_steps, train_accuracies, linestyle='-', color='blue', label='Train Accuracy')
    axs[0].plot(update_steps, val_accuracies, linestyle='-', color='red', label='Validation Accuracy')
    axs[0].set_ylim(0, None)
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Update Steps')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(update_steps, train_losses, linestyle='-', color='blue', label='Train Loss')
    axs[1].plot(update_steps, val_losses, linestyle='-', color='red', label='Validation Loss')
    axs[1].set_ylim(0, None)
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Update Steps')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    axs[2].plot(update_steps, train_costs, linestyle='-', color='blue', label='Train Cost')
    axs[2].plot(update_steps, val_costs, linestyle='-', color='red', label='Validation Cost')
    axs[2].set_ylim(0, None)
    axs[2].set_title('Cost')
    axs[2].set_xlabel('Update Steps')
    axs[2].set_ylabel('Cost')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


class Cifar10: 
    def __init__(self, train_size, val_size):
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

        (X_train, y_train), (X_test, y_test) = \
            keras.datasets.cifar10.load_data()
        
        # Images.
        self.X_train = X_train[:train_size]
        self.X_val = X_train[50000-val_size:]
        self.X_test = X_test

        # Labels.
        self.y_train = y_train[:train_size]
        self.y_val = y_train[50000-val_size:]
        self.y_test = y_test

        self.X_train_std, self.Y_train, self.X_val_std, self.Y_val, self.X_test_std, self.Y_test = \
            self.preprocess()

        
    def preprocess(self):
        """Preprocesses the CIFAR-10 dataset."""
        # Convert type: uint8 -> float64
        X_train_std = self.X_train.astype('float64')
        X_val_std = self.X_val.astype('float64')
        X_test_std = self.X_test.astype('float64')

        # Normalize values: 0-255 -> 0-1
        X_train_std = X_train_std / 255.0
        X_val_std = X_val_std / 255.0
        X_test_std = X_test_std / 255.0

        # Flatten images: 50000x32x32x3 -> 50000x3072
        X_train_std = X_train_std.reshape(X_train_std.shape[0], -1)
        X_val_std = X_val_std.reshape(X_val_std.shape[0], -1)
        X_test_std = X_test_std.reshape(X_test_std.shape[0], -1)

        # One-hot encode labels: 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Y_train = np.zeros((self.y_train.shape[0], 10))
        Y_train[np.arange(self.y_train.shape[0]), self.y_train.flatten()] = 1
        Y_val = np.zeros((self.y_val.shape[0], 10))
        Y_val[np.arange(self.y_val.shape[0]), self.y_val.flatten()] = 1
        Y_test = np.zeros((self.y_test.shape[0], 10))
        Y_test[np.arange(self.y_test.shape[0]), self.y_test.flatten()] = 1

        # Standardize data to have a mean of 0 and standard deviation of 1
        means = np.mean(X_train_std, axis=0).reshape(1, -1)
        standard_devs = np.std(X_train_std, axis=0).reshape(1, -1)
        X_train_std = (X_train_std - means) / standard_devs
        X_val_std = (X_val_std - means) / standard_devs
        X_test_std = (X_test_std - means) / standard_devs
        
        # Transpose data to match assignment description and lectures
        X_train_std = X_train_std.T
        Y_train = Y_train.T
        X_val_std = X_val_std.T
        Y_val = Y_val.T
        X_test_std = X_test_std.T
        Y_test = Y_test.T

        return X_train_std, Y_train, X_val_std, Y_val, X_test_std, Y_test
    

    def get_data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    

    def get_preprocessed_data(self):
        return self.X_train_std, self.Y_train, self.X_val_std, self.Y_val, self.X_test_std, self.Y_test


class Layer:
    def __init__(self, input_size, output_size, batch_norm=False):
        self.w = self.generate_weights((output_size, input_size))
        self.b = self.generate_biases((output_size, 1))
        if batch_norm:
            self.gamma = np.ones((output_size, 1))
            self.beta = np.zeros((output_size, 1))

    
    def generate_weights(self, shape):
        """Generates weights with Gaussian random values."""
        return np.random.normal(loc=0.0, scale=math.sqrt(2.0/shape[1]), size=shape)


    def generate_biases(self, shape):
        """Generates biases with zeros."""
        return np.zeros(shape)


class KLayerNeuralNetwork:
    def __init__(self, layer_sizes, batch_norm):
        self.layers = []
        for input_size, output_size in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            self.layers.append(Layer(input_size, output_size, batch_norm=batch_norm))
        self.layers.append(Layer(layer_sizes[-2], layer_sizes[-1]))
        self.batch_norm = batch_norm
        
        # Used for eval.
        self.update_steps = []
        self.train_accuracies = []
        self.train_costs = []
        self.train_losses = []
        self.val_accuracies = []
        self.val_costs = []
        self.val_losses = []

    
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
        sum = 0.0
        for layer in self.layers:
            sum += np.sum(layer.w ** 2)
        return sum
    

    def compute_cost(self, X_std, Y, λ):
        """Compute the cost function."""
        P = None
        if self.batch_norm:
            _, _, _, P, _, _ = self.evaluate_classifier(X_std)
        else:
            P = self.evaluate_classifier(X_std)[-1]
        return self.cross_entropy_loss(Y, P) + λ*self.l2_regularization()
    

    def compute_accuracy(self, X_std, y):
        """Compute the accuracy of the model."""
        P = None
        if self.batch_norm:
            _, _, _, P, _, _ = self.evaluate_classifier(X_std)
        else:
            P = self.evaluate_classifier(X_std)[-1]
        return np.sum(np.argmax(P, axis=0) == y.flatten()) / P.shape[1]
    

    def compute_metrics(self, X_std, Y, y, λ):
        """Compute accuracy, cost and loss of the model."""
        accuracy = self.compute_accuracy(X_std, y)
        cost = self.compute_cost(X_std, Y, λ)
        P = None
        if self.batch_norm:
            _, _, _, P, _, _ = self.evaluate_classifier(X_std)
        else:
            P = self.evaluate_classifier(X_std)[-1]
        loss = self.cross_entropy_loss(Y, P)
        return accuracy, cost, loss
    

    def update_metrics(self, X_train_std, Y_train, y_train, X_val_std, Y_val, y_val, λ, t):
        """Print the metrics for train and val data and save it into lists for later use."""
        self.update_steps.append(t)

        train_accuracy, train_cost, train_loss = self.compute_metrics(X_train_std, Y_train, y_train, λ)
        self.train_accuracies.append(train_accuracy)
        self.train_costs.append(train_cost)
        self.train_losses.append(train_loss)

        val_accuracy, val_cost, val_loss = self.compute_metrics(X_val_std, Y_val, y_val, λ)
        self.val_accuracies.append(val_accuracy)
        self.val_costs.append(val_cost)
        self.val_losses.append(val_loss)

        print()
        print(f'update_step = {t}')
        print(f' - training data:    acc {train_accuracy:.4f}, cost {train_cost:.4f} loss {train_loss:.4f}')
        print(f' - validation data:  acc {val_accuracy:.4f}, cost {val_cost:.4f} loss {val_loss:.4f}')
    

    def batch_normalise(self, S, mean, variance):
        return np.diag(np.power(variance.flatten() + 1e-6, -0.5)) @ (S - mean)


    def evaluate_classifier(self, X_std):
        """Compute the forward pass."""
        if self.batch_norm:
            S_list = []
            S_hat_list = []
            X_list = []
            mean_list = []
            variance_list = []
            X = X_std
            for layer in self.layers[:-1]:
                S = layer.w @ X + layer.b # 12
                S_list.append(S)
                mean = np.mean(S, axis=1).reshape(-1, 1) # 13
                mean_list.append(mean)
                variance = np.var(S, axis=1).reshape(-1, 1) # 14
                variance_list.append(variance)
                S_hat = self.batch_normalise(S, mean, variance) # 15
                S_hat_list.append(S_hat)
                S_squiggle = layer.gamma * S_hat + layer.beta # 16
                X = self.relu(S_squiggle) # 17
                X_list.append(X)
            P = self.softmax(self.layers[-1].w @ X + self.layers[-1].b) # 18, 19
            return S_list, S_hat_list, X_list, P, mean_list, variance_list
        else:
            S = []
            prev_X = X_std
            for layer in self.layers[:-1]:
                prev_X = self.relu(layer.w @ prev_X + layer.b)
                S.append(prev_X)
            S.append(self.softmax(self.layers[-1].w @ prev_X + self.layers[-1].b))
            return S
        

    def batch_norm_backward(self, G, S, mean, variance):
        """Compute the backward pass for batch norm."""
        n = G.shape[1]
        sigma1 = np.power(variance + 1e-6, -0.5) # 31
        sigma2 = np.power(variance + 1e-6, -1.5) # 32 
        G1 = G * sigma1 # 33
        G2 = G * sigma2 # 34
        D = S - mean # 35
        c = (G2 * D) @ np.ones((n, 1)) # 36
        return G1 - np.mean(G1, axis=1).reshape(-1, 1) - (1/n)*D*c # 37
    
    
    def compute_gradients(self, X_std, Y, λ):
        """Compute the backward pass."""
        if self.batch_norm:
            n = X_std.shape[1]
            S_list, S_hat_list, X_list, P, mean_list, variance_list = \
                self.evaluate_classifier(X_std)
            G = -(Y - P) # 21
            grad_wk = (1/n) * G @ X_list[-1].T + (2*λ) * self.layers[-1].w
            grad_bk = np.mean(G, axis=1).reshape(-1, 1) # 22
            grads = [(grad_wk, grad_bk)]
            G = self.layers[-1].w.T @ G # 23
            G = G * np.where(X_list[-1] > 0, 1, 0) # 24
            for i in range(len(self.layers)-2, 0, -1):
                grad_gamma = np.mean(G * S_hat_list[i], axis=1).reshape(-1, 1)
                grad_beta = np.mean(G, axis=1).reshape(-1, 1) # 25
                G = G * self.layers[i].gamma # 26
                G = self.batch_norm_backward(G, S_list[i], mean_list[i], variance_list[i]) # 27
                grad_w = (1/n) * G @ X_list[i-1].T + (2*λ) * self.layers[i].w
                grad_b = np.mean(G, axis=1).reshape(-1, 1) # 28
                G = self.layers[i].w.T @ G # 29
                G = G * np.where(X_list[i-1] > 0, 1, 0) # 30
                grads.append((grad_w, grad_b, grad_gamma, grad_beta))
            # Compute for layer 1
            grad_gamma = np.mean(G * S_hat_list[0], axis=1).reshape(-1, 1)
            grad_beta = np.mean(G, axis=1).reshape(-1, 1)
            G = G * self.layers[0].gamma
            G = self.batch_norm_backward(G, S_list[0], mean_list[0], variance_list[0])
            grad_w = (1/n) * G @ X_std.T + (2*λ) * self.layers[0].w
            grad_b = np.mean(G, axis=1).reshape(-1, 1)
            grads.append((grad_w, grad_b, grad_gamma, grad_beta))
            return grads
        else:
            batch_size = X_std.shape[1]
            scores = self.evaluate_classifier(X_std)
            P = scores[-1]
            G = -(Y - P)
            grads = []
            for layer, S in zip(reversed(self.layers[1:]), reversed(scores[:-1])):
                grad_w = (1 / batch_size) * G @ S.T + (2*λ) * layer.w
                grad_b = np.mean(G, axis=1).reshape(-1, 1)
                grads.append((grad_w, grad_b))
                G = layer.w.T @ G
                G = G * np.where(S > 0, 1, 0)
            grad_w1 = (1 / batch_size) * G @ X_std.T + (2*λ) * self.layers[0].w
            grad_b1 = np.mean(G, axis=1).reshape(-1, 1)
            grads.append((grad_w1, grad_b1))
            return reversed(grads)
    

    def update_parameters(self, grads, learning_rate):
        """Update the models parameters."""
        if self.batch_norm:
            for i, (grad_w, grad_b, grad_gamma, grad_beta) in enumerate(reversed(grads[1:])):
                w = self.layers[i].w
                b = self.layers[i].b
                gamma = self.layers[i].gamma
                beta = self.layers[i].beta
                w -= learning_rate * grad_w
                b -= learning_rate * grad_b
                gamma -= learning_rate * grad_gamma
                beta -= learning_rate * grad_beta
            w = self.layers[-1].w
            b = self.layers[-1].b
            w -= learning_rate*grads[0][0]
            b -= learning_rate*grads[0][1]
        else:
            for i, (grad_w, grad_b) in enumerate(grads):
                w = self.layers[i].w
                b = self.layers[i].b
                w -= learning_rate * grad_w
                b -= learning_rate * grad_b


    def train(self, X_train_std, Y_train, y_train, X_val_std, Y_val, y_val, learning_rates, batch_size, λ, train_size):
        indices = np.arange(X_train_std.shape[1])
        
        for t, learning_rate in enumerate(learning_rates):
            if t * batch_size % train_size == 0:
                np.random.shuffle(indices)
                X_shuffled = X_train_std[:, indices]
                Y_shuffled = Y_train[:, indices]

            if t % 100 == 0:
                self.update_metrics(X_train_std, Y_train, y_train, X_val_std, Y_val, y_val, λ, t)

            start = ((t)*batch_size) % train_size
            end = min(start+batch_size, X_train_std.shape[1])

            grads = self.compute_gradients(X_shuffled[:, start:end], Y_shuffled[:, start:end], λ)
            self.update_parameters(grads, learning_rate)

        t = len(learning_rates)
        self.update_metrics(X_train_std, Y_train, y_train, X_val_std, Y_val, y_val, λ, t)
        
        return self.update_steps, self.train_accuracies, self.train_losses, self.train_costs, self.val_accuracies, self.val_losses, self.val_costs


def main():
    # Set hyperparameters.
    eta_min = 1e-5
    eta_max = 1e-1
    ns = 2 * (45000 // 100)
    num_cycles = 2
    learning_rates = KLayerNeuralNetwork.generate_cyclical_learning_rates(eta_min, eta_max, ns, num_cycles)
    batch_size = 100
    λ = 0.0035
    train_size = 45000
    val_size = 5000
    batch_norm = True

    # Get data.
    cifar10 = Cifar10(train_size, val_size)
    labels = cifar10.labels
    X_train, y_train, X_val, y_val, X_test, y_test = cifar10.get_data()
    X_train_std, Y_train, X_val_std, Y_val, X_test_std, Y_test = cifar10.get_preprocessed_data()
    
    # Train network.
    nn = KLayerNeuralNetwork([3072, 70, 30, 10], batch_norm)
    update_steps, train_accuracies, train_losses, train_costs, val_accuracies, val_losses, val_costs = \
        nn.train(X_train_std, Y_train, y_train, X_val_std, Y_val, y_val, learning_rates, batch_size, λ, train_size)
    
    # Evaluate network on test data.
    test_accuracy, test_cost, test_loss = nn.compute_metrics(X_test_std, Y_test, y_test, λ)
    print()  
    print('Final results on test data:')  
    print(f' - accuracy {test_accuracy:.4f}')
    print(f' - cost     {test_cost:.4f}')
    print(f' - loss     {test_loss:.4f}')
    plot(update_steps, train_accuracies, train_losses, train_costs, val_accuracies, val_losses, val_costs)


if __name__ == "__main__":
    main()
