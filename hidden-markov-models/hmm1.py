def read_matrix():
    tokens = input().strip().split()

    num_rows, num_cols = int(tokens[0]), int(tokens[1])

    numbers = [float(x) for x in tokens[2:]]

    matrix = []
    for row in range(num_rows):
        row = numbers[row * num_cols:(row + 1) * num_cols]
        matrix.append(row)

    return matrix

def read_observation_sequence():
    return [int(x) for x in input().strip().split()[1:]]

def alpha_pass(A, B, π, o):
    N = len(A)
    T = len(o)

    Alpha = [[0 for _ in range(N)] for _ in range(T)]
    for state in range(N):
        Alpha[0][state] = π[state] * B[state][o[0]]
    
    for t in range(1, T):
        for i in range(N):
            Alpha[t][i] = sum(A[j][i] * Alpha[t-1][j] for j in range(N)) * B[i][o[t]]

    print(sum(Alpha[T-1]))

def main():
    A = read_matrix() # State transition matrix
    B = read_matrix() # Observation matrix
    π = read_matrix() # Initial state probability distribution
    o = read_observation_sequence() # Observation sequence

    alpha_pass(A, B, π[0], o)
    
if __name__ == "__main__":
    main()