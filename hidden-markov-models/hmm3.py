import math

def read_matrix():
    """
    Reads in one matrix from standard input and returns it as a matrix of floating
    point numbers. 
    """

    tokens = input().strip().split()

    rows, cols = int(tokens[0]), int(tokens[1])
    numbers = [float(x) for x in tokens[2:]]

    matrix = []
    for row in range(rows):
        matrix.append(numbers[row * cols:(row + 1) * cols])

    return matrix



def read_observation_sequence():
    """
    Reads the sequence of observations from standard input and returns it as a list of
    integers.
    """

    return [int(x) for x in input().strip().split()[1:]]



def print_matrix(matrix):
    """
    Print matrix to standard output. First two numbers on the line are the number of rows and 
    columns respectively, followed by all the elements in the matrix in row major order.
    """

    rows = len(matrix)
    cols = len(matrix[0])
    numbers = ' '.join(str(matrix[row][col]) for row in range(rows) for col in range(cols))
    print(rows, cols, numbers)



def alpha_pass(A, B, π, o):
    """
    Computes the alpha pass and returns the entire matrix of alpha values. This version
    uses scaling to avoid multiplications of probabilities resulting in 0.
    """

    N = len(A)
    T = len(o)

    Alpha = [[0 for _ in range(N)] for _ in range(T)]
    scaling_factor = [0 for _ in range(T)]

    # Compute alpha_0
    for i in range(N):
        Alpha[0][i] = π[i] * B[i][o[0]]
        scaling_factor[0] += Alpha[0][i]

    # Scale alpha_0
    scaling_factor[0] = 1/scaling_factor[0]
    for i in range(N):
        Alpha[0][i] = scaling_factor[0] * Alpha[0][i]

    # Compute alpha_t
    for t in range(1, T):
        for i in range(N):
            Alpha[t][i] = sum(Alpha[t-1][j] * A[j][i] for j in range(N)) * B[i][o[t]]
            scaling_factor[t] += Alpha[t][i]

        # Scale alpha_t
        scaling_factor[t] = 1/scaling_factor[t]
        for i in range(N):
            Alpha[t][i] = scaling_factor[t] * Alpha[t][i]

    return Alpha, scaling_factor



def beta_pass(A, B, o, scaling_factor):
    """
    Computes the beta pass and returns the entire matrix of beta values. 
    This version uses the same scaling factor as the alpha-pass.
    """

    N = len(A)
    T = len(o)

    Beta = [[0 for _ in range(N)] for _ in range(T)]

    # Compute beta_T-1 (1 * scaling factor)
    for i in range(N):
        Beta[T-1][i] = scaling_factor[T-1]

    # Compute beta_t
    for t in range(T-2, -1, -1):
        for i in range(N):
            Beta[t][i] = sum(A[i][j] * B[j][o[t+1]] * Beta[t+1][j] for j in range(N))

            # Scale beta_t
            Beta[t][i] = scaling_factor[t] * Beta[t][i]

    return Beta



def gamma_di_gamma(A, B, Alpha, Beta, o):
    """
    Calculates the probabilities of being in state i at timestep t and transitioning to 
    timestep j at timestep t+1. Returns a 3D matrix representation. We do not need to
    normalize since Alpha and Beta are already scaled.
    """

    N = len(A)
    T = len(o)

    Gamma = [[0 for _ in range(N)] for _ in range(T)]
    Di_gamma = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(T)]

    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                Di_gamma[t][i][j] = Alpha[t][i] * A[i][j] * B[j][o[t+1]] * Beta[t+1][j]
                Gamma[t][i] += Di_gamma[t][i][j]

    # Special case for T-1
    for i in range(N):
        Gamma[T-1][i] = Alpha[T-1][i]

    return Gamma, Di_gamma



def re_estimate(A, B, π, o, Gamma, Di_gamma):
    """
    Re-estimates the hidden markov model's parameters. 
    """

    M = len(B[0])
    N = len(A)
    T = len(o)

    # Re-estimate π
    for i in range(N):
        π[i] = Gamma[0][i]

    # Re-estimate A
    for i in range(N):
        denominator = 0
        for t in range(T-1):
            denominator += Gamma[t][i]
        
        for j in range(N):
            numerator = 0
            for t in range(T-1):
                numerator += Di_gamma[t][i][j]
            A[i][j] = numerator / denominator

    # Re-estimate B
    for i in range(N):
        denominator = 0
        for t in range(T):
            denominator += Gamma[t][i]
        
        for j in range(M):
            numerator = 0
            for t in range(T):
                if o[t] == j:
                    numerator += Gamma[t][i]
            B[i][j] = numerator / denominator

    return A, B, π



def compute_log_prob(scaling_factor, T):
    """
    Compute the logarithmic probability.
    """

    log_prob = 0

    for i in range(T):
        log_prob += math.log(scaling_factor[i])

    return -log_prob



def main():
    A = read_matrix() # Initial guess of state transition matrix
    B = read_matrix() # Initial guess of observation matrix
    π = read_matrix()[0] # Initial guess of intitial state probability distribution
    o = read_observation_sequence() # Observation sequence

    max_iterations = 30
    old_log_prob = float('-inf')

    # Baum-Welch
    for _ in range(max_iterations):
        Alpha, scaling_factor = alpha_pass(A, B, π, o)
        Beta = beta_pass(A, B, o, scaling_factor)
        Gamma, Di_gamma = gamma_di_gamma(A, B, Alpha, Beta, o)
        A, B, π = re_estimate(A, B, π, o, Gamma, Di_gamma)

        log_prob = compute_log_prob(scaling_factor, len(o))
        if log_prob > old_log_prob:
            old_log_prob = log_prob
        else:
            break

    print_matrix(A)
    print_matrix(B)

if __name__ == "__main__":
    main()