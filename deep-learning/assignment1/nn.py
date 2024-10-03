import keras
import matplotlib.pyplot as plt
import numpy as np


def plot_image_with_probabilities(X, X_std, y, labels, index, W, b):
    probabilities = evaluate_classifier(X_std[:, index:index+1], W, b)
    predicted_label = np.argmax(probabilities)
    image = X[index].reshape((32, 32, 3))
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Actual: {labels[y[index, 0]]}, Predicted: {labels[predicted_label]}")
    plt.subplot(1, 2, 2)
    plt.bar(range(len(labels)), probabilities.flatten())
    plt.xticks(range(len(labels)), labels.values(), rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Predicted Probabilities')
    plt.tight_layout()
    plt.show()


def plot_images_with_probabilities(X, X_std, y, labels, W, b):
    indices = np.random.choice(X.shape[0], size=9, replace=False)
    _, axes = plt.subplots(3, 6, figsize=(18, 9))
    for i, index in enumerate(indices):
        probabilities = evaluate_classifier(X_std[:, index:index+1], W, b)
        predicted_label = np.argmax(probabilities)
        image = X[index].reshape((32, 32, 3))
        ax_image = axes[i // 3, 2 * (i % 3)]
        ax_image.imshow(image)
        ax_image.axis('off')
        ax_image.set_title(f"Actual: {labels[y[index, 0]]}, Predicted: {labels[predicted_label]}")
        ax_prob = axes[i // 3, 2 * (i % 3) + 1] 
        ax_prob.bar(range(len(labels)), probabilities.flatten())
        ax_prob.set_xticks(range(len(labels)))
        ax_prob.set_xticklabels(labels.values(), rotation=45)  
        ax_prob.set_ylabel('Probability') 
    plt.tight_layout()
    plt.show()


def plot_weights(W, labels):
    _, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        im = np.reshape(W[i, :], (32, 32, 3))
        im_normalized = (im - np.min(im)) / (np.max(im) - np.min(im))
        im_permuted = np.transpose(im_normalized, (1, 0, 2))
        row = i // 5
        col = i % 5
        axes[row, col].imshow(im_permuted)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'{labels[i]}')
    plt.tight_layout()
    plt.show()


def plot_costs(costs):
    plt.plot(costs, marker='o', linestyle='-')
    plt.title('Costs over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()


def plot_losses(losses):
    plt.plot(losses, marker='o', linestyle='-')
    plt.title('Losses over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def plot_accuracies(accuracies):
    plt.plot(accuracies, marker='o', linestyle='-')
    plt.title('Accuracies over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()


def preprocess(X_train, y_train, X_test, y_test):
    """Preprocesses the CIFAR-10 dataset."""
    # Convert type: uint8 -> float64
    # Normalize values: 0-255 -> 0-1
    X_train_norm = X_train.astype('float64') / 255.0
    X_test_norm = X_test.astype('float64') / 255.0

    # Flatten images: 32x32x3 -> 3072
    X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], -1)
    X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], -1)

    # One-hot encode labels: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train_onehot = np.zeros((50000, 10))
    y_train_onehot[np.arange(50000), y_train.flatten()] = 1
    y_test_onehot = np.zeros((10000, 10))
    y_test_onehot[np.arange(10000), y_test.flatten()] = 1

    # Standardize input data to have a mean of 0 and standard deviation of 1
    train_mean_array = np.mean(X_train_norm, axis=0).reshape(1, -1)
    train_std_array = np.std(X_train_norm, axis=0).reshape(1, -1)
    test_mean_array = np.mean(X_test_norm, axis=0).reshape(1, -1)
    test_std_array = np.std(X_test_norm, axis=0).reshape(1, -1)
    X_train_std = (X_train_norm - train_mean_array) / train_std_array
    X_test_std = (X_test_norm - test_mean_array) / test_std_array

    # Transpose data to match assignment description
    X_train_std = X_train_std.T
    X_test_std = X_test_std.T
    y_train_onehot = y_train_onehot.T
    y_test_onehot = y_test_onehot.T

    return X_train_std, y_train_onehot, X_test_std, y_test_onehot


def generate_weights():
    """Generates weights with Gaussian random values."""
    return np.random.normal(loc=0, scale=0.01, size=(10, 3072))


def generate_biases():
    """Generates biases with Gaussian random values."""
    return np.random.normal(loc=0, scale=0.01, size=(10, 1))


def dense_layer(X, W, b):
    """Compute WX + b."""
    return W @ X + b


def softmax(s):
    """Compute the softmax activation function."""
    exp_s = np.exp(s)
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)


def evaluate_classifier(X, W, b):
    """Evaluate the classifier on a batch of input images."""
    return softmax(dense_layer(X, W, b))


def cross_entropy_loss(Y, P):
    """Compute the average cross-entropy loss for a batch."""
    return np.sum(-np.sum(Y * np.log(P), axis=0)) / Y.shape[1]


def l2_regularization(W):
    """Compute L2 regularization for the given weight matrix W."""
    return np.sum(W ** 2)


def compute_cost(X, Y, W, b, λ):
    """Compute the cost function."""
    P = evaluate_classifier(X, W, b)
    return cross_entropy_loss(Y, P) + λ * l2_regularization(W)


def compute_accuracy(X, y, W, b):
    """Compute the accuracy of the model."""
    P = evaluate_classifier(X, W, b)
    return np.sum(np.argmax(P, axis=0) == y.flatten()) / P.shape[1]


def compute_gradients(X, Y, W, b, λ):
    """Compute the gradients of the loss function."""
    n = X.shape[1]
    P = evaluate_classifier(X, W, b)
    G = -(Y - P)
    grad_W = (1/n) * G @ X.T + (2*λ) * W
    grad_b = (1/n) * G @ np.ones((n, 1))
    return grad_W, grad_b


def compute_gradients_num(X, Y, W, b, lamda, h):
	"""Copied from functions.py (only used in development)"""
	no 	= 	W.shape[0]
	
	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	c = compute_cost(X, Y, W, b, lamda)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = compute_cost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = compute_cost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]


def mini_batch_gradient_descent(epochs, learning_rate, batch_size, X, Y, W, b, λ, X_test, Y_test, y, y_test):
    """Perform mini batch gradient descent."""
    indices = np.arange(X.shape[1])
    train_losses = []
    train_costs = []
    train_accuracies = []
    test_losses = []
    test_costs = []
    test_accuracies = []
    for epoch in range(epochs):
        np.random.shuffle(indices)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]

        P_train = evaluate_classifier(X, W, b)
        train_loss = cross_entropy_loss(Y, P_train)
        P_test = evaluate_classifier(X_test, W, b)
        test_loss = cross_entropy_loss(Y_test, P_test)

        train_cost = compute_cost(X, Y, W, b, λ)
        test_cost = compute_cost(X_test, Y_test, W, b, λ)

        train_accuracy = compute_accuracy(X, y, W, b)
        test_accuracy = compute_accuracy(X_test, y_test, W, b)

        train_losses.append(train_loss)
        train_costs.append(train_cost)
        train_accuracies.append(train_accuracy)

        test_losses.append(test_loss)
        test_costs.append(test_cost)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{epochs} completed.")
        print(f" - Loss on test data: {test_loss}")
        print(f" - Cost on test data: {test_cost}")
        print(f" - Accuracy on test data: {test_accuracy}")

        for start in range(0, X.shape[1], batch_size):
            end = min(start + batch_size, X.shape[1])

            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]

            W_grad, b_grad = compute_gradients(X_batch, Y_batch, W, b, λ)

            W = W - learning_rate * W_grad
            b = b - learning_rate * b_grad

    return W, b, train_losses, train_costs, train_accuracies, test_losses, test_costs, test_accuracies


def main():
    labels = {
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

    # Load and preprocess data.
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train_std, y_train_onehot, X_test_std, y_test_onehot = preprocess(X_train, y_train, X_test, y_test)

    # Generate learnable parameters.
    W = generate_weights()
    b = generate_biases() 

    # Set hyperparameters.
    epochs = 40
    learning_rate = 0.001
    batch_size = 100
    λ = 1.0
   
    # Train.
    W, b, train_losses, train_costs, train_accuracies, test_losses, test_costs, test_accuracies = \
        mini_batch_gradient_descent(epochs, learning_rate, batch_size, X_train_std, y_train_onehot, W, b, λ, X_test_std, y_test_onehot, y_train, y_test)

    # Final results.
    P_train = evaluate_classifier(X_train_std, W, b)
    train_loss = cross_entropy_loss(y_train_onehot, P_train)
    P_test = evaluate_classifier(X_test_std, W, b)
    test_loss = cross_entropy_loss(y_test_onehot, P_test)
    train_cost = compute_cost(X_train_std, y_train_onehot, W, b, λ)
    train_accuracy = compute_accuracy(X_train_std, y_train, W, b)
    test_cost = compute_cost(X_test_std, y_test_onehot, W, b, λ)
    test_accuracy = compute_accuracy(X_test_std, y_test, W, b)

    train_costs.append(train_cost)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_costs.append(test_cost)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print("#---------------------------------------------------#")
    print("Score on training data:")
    print(f" - Cost: {train_cost}")
    print(f" - Accuracy: {train_accuracy}")
    print("Score on test data:")
    print(f" - Cost: {test_cost}")
    print(f" - Accuracy: {test_accuracy}")

    # Uncomment to plot images. 
    # plot_image_with_probabilities(X_train, X_train_std, y_train, labels, 0, W, b)
    # plot_images_with_probabilities(X_train, X_train_std, y_train, labels, W, b)
    # plot_weights(W, labels)
    # plot_costs(train_costs)
    # plot_costs(test_costs)
    # plot_losses(train_losses)
    # plot_losses(test_losses)
    # plot_accuracies(train_accuracies)
    # plot_accuracies(test_accuracies)
    
 
if __name__ == "__main__":
    main()
