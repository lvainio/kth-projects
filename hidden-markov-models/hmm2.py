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

def arg_max(l):
    return max(range(len(l)), key=l.__getitem__)

def viterbi(A, B, π, o):
    N = len(A)
    T = len(o)

    delta = [[0 for _ in range(N)] for _ in range(T)]
    delta_idx = [[0 for _ in range(N)] for _ in range(T)]

    # Initial values.
    for state in range(N):
        delta[0][state] = π[0][state] * B[state][o[0]]        

    # Going through each time step.
    for t in range(1, T):
        for state in range(N):
            max_probability = float('-inf')
            max_state = -1
            for prev in range(N):
                probability = A[prev][state] * delta[t-1][prev] * B[state][o[t]]
                if probability > max_probability:
                    max_probability = probability
                    max_state = prev

            delta[t][state] = max_probability
            delta_idx[t][state] = max_state

    sequence = [0 for _ in range(T)]
    sequence[T-1] = arg_max(delta[T-1])

    # Backtrack
    for t in range(T-2, -1, -1):
        sequence[t] = delta_idx[t+1][sequence[t+1]]
    
    print(' '.join(str(state) for state in sequence))

def main():
    A = read_matrix() # State transition matrix
    B = read_matrix() # Observation matrix
    π = read_matrix() # Initial state probability distribution
    o = read_observation_sequence() # Observation sequence

    viterbi(A, B, π, o)

if __name__ == "__main__":
    main()