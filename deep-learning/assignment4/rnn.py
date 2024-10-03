import numpy as np

# Data
with open('goblet_of_fire.txt', 'r', encoding='utf-8') as file:
    data = file.read()
characters = sorted(set(data))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

# Hyperparameters
m = 100
k = len(characters)
eta = 0.1
seq_length = 25
sigma = 0.01
epochs = 1

# Trainable parameters
b = np.zeros((m, 1))
c = np.zeros((k, 1))
U = np.random.normal(loc=0, scale=sigma, size=(m, k))
V = np.random.normal(loc=0, scale=sigma, size=(k, m))
W = np.random.normal(loc=0, scale=sigma, size=(m, m))

# AdaGrad accumulated squared gradients
acc_grad_b = np.zeros((m, 1))
acc_grad_c = np.zeros((k, 1))
acc_grad_U = np.zeros((m, k))
acc_grad_V = np.zeros((k, m))
acc_grad_W = np.zeros((m, m))

def softmax(o):
    exp_o = np.exp(o)
    return exp_o / np.sum(exp_o)

def cross_entropy_loss(y, p):
    return -np.log(p[np.argmax(y)])

def forward(X, Y, h_prev):
    a_list = []
    h_list = []
    p_list = []
    loss = 0
    h = h_prev
    for t in range(X.shape[1]): 
        x = X[:, t].reshape(-1, 1)
        y = Y[:, t].reshape(-1, 1)
        a = W @ h + U @ x + b # 1
        h = np.tanh(a) # 2
        o = V @ h + c # 3
        p = softmax(o) # 4 
        a_list.append(a)
        h_list.append(h)
        p_list.append(p)
        loss += cross_entropy_loss(y, p)
    return a_list, h_list, p_list, loss

def backward(X, Y, h_prev, a_list, h_list, p_list):
    grad_o = [(-(Y[:, t].reshape(-1, 1) - p)).T for t, p in enumerate(p_list)]
    grad_h = [None] * 24 + [grad_o[-1] @ V]
    grad_a = [None] * 24 + [grad_h[-1] @ np.diag((1 - (np.tanh(a_list[-1])**2)).flatten())]
    for t in range(len(a_list)-2, -1, -1):
        grad_h[t] = grad_o[t] @ V + grad_a[t+1] @ W
        grad_a[t] = grad_h[t] @ np.diag((1 - (np.tanh(a_list[t])**2)).flatten())
    grad_b = np.clip(sum(da.T for da in grad_a), -5, 5)
    grad_c = np.clip(sum(do.T for do in grad_o), -5, 5)
    grad_U = np.clip(sum(grad_a[t].T @ X[:, t].reshape(-1, 1).T for t in range(len(a_list))), -5, 5)
    grad_V = np.clip(sum(do.T @ h.T for do, h in zip(grad_o, h_list)), -5, 5)
    grad_W = np.clip(grad_a[0] @ h_prev + sum(da.T @ h.T for da, h in zip(grad_a[1:], h_list[:-1])), -5, 5)
    return grad_b, grad_c, grad_U, grad_V, grad_W

def update_parameters(grad_b, grad_c, grad_U, grad_V, grad_W):
    global b, c, U, V, W
    global acc_grad_b, acc_grad_c, acc_grad_U, acc_grad_V, acc_grad_W
    acc_grad_b += np.square(grad_b)
    acc_grad_c += np.square(grad_c)
    acc_grad_U += np.square(grad_U)
    acc_grad_V += np.square(grad_V)
    acc_grad_W += np.square(grad_W)
    b -= (eta / np.sqrt(acc_grad_b + 1e-8)) * grad_b
    c -= (eta / np.sqrt(acc_grad_c + 1e-8)) * grad_c
    U -= (eta / np.sqrt(acc_grad_U + 1e-8)) * grad_U
    V -= (eta / np.sqrt(acc_grad_V + 1e-8)) * grad_V
    W -= (eta / np.sqrt(acc_grad_W + 1e-8)) * grad_W


def one_hot_encode(index):
    """"""
    one_hot = np.zeros((80, 1))
    one_hot[index] = 1
    return one_hot

def one_hot_encode_chars(chars):
    """takes in a (n,) numpy array of chars and returns (k, n) one hot matrix"""
    one_hot_matrix = np.zeros((k, len(chars)))
    for i, char in enumerate(chars):
        index = char_to_index[char]
        one_hot_matrix[index, i] = 1
    return one_hot_matrix

def synthesize(x1, n, h_prev):
    """"""
    h_prev = h_prev
    x = x1
    indices = []
    for _ in range(n):
        a = W @ h_prev + U @ x + b # 1
        h = np.tanh(a) # 2
        o = V @ h + c # 3
        p = softmax(o) # 4

        char_indices = np.arange(len(p))
        char_index = np.random.choice(char_indices, p=p.reshape(-1))

        indices.append(char_index)
        x = one_hot_encode(char_index)
        
    return ''.join(index_to_char[i] for i in indices)


# Training
update_steps = 0
for epoch in range(epochs):

    print("---------------------")
    print("EPOCH {epoch}")
    print("---------------------")

    h_prev = np.zeros((m, 1))
    e = 0
    smooth_loss = None

    while e < len(data) - seq_length - 1:
        
        x_chars = np.array(list(data[e:e+seq_length]))
        y_chars = np.array(list(data[e+1:e+seq_length+1]))
        X = one_hot_encode_chars(x_chars)
        Y = one_hot_encode_chars(y_chars)

        a_list, h_list, p_list, loss = forward(X, Y, h_prev)

        if smooth_loss is None:
            smooth_loss = loss
        else:
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss

        grad_b, grad_c, grad_U, grad_V, grad_W = backward(X, Y, h_prev, a_list, h_list, p_list)

        update_parameters(grad_b, grad_c, grad_U, grad_V, grad_W)

        if update_steps % 100 == 0:
            print(f"update: {update_steps}, loss: {smooth_loss}")

        if update_steps % 500 == 0:
            text = synthesize(X[:, 0].reshape(-1, 1), 200, h_prev)
            print("Synthesized text:\n", text)


        h_prev = h_list[-1]
        e += seq_length
        update_steps += 1
