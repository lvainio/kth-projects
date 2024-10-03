def read_matrix():
    tokens = input().strip().split()

    num_rows, num_cols = int(tokens[0]), int(tokens[1])

    numbers = [float(x) for x in tokens[2:]]

    matrix = []
    for row in range(num_rows):
        row = numbers[row * num_cols:(row + 1) * num_cols]
        matrix.append(row)

    return matrix

def mat_mul(A, B):
    res_rows = len(A)
    res_cols = len(B[0])

    res = [[0 for _ in range(res_cols)] for _ in range(res_rows)]
    
    for row in range(len(A)):
        for col in range(len(B[0])):
            for i in range(len(B)):
                res[row][col] += A[row][i] * B[i][col]

    return res

def main():
    A = read_matrix() # State transition matrix
    B = read_matrix() # Observation matrix
    π = read_matrix() # Initial state probability distribution

    π = mat_mul(π, A) 
    obs_prob = mat_mul(π, B) 

    rows, cols = len(obs_prob), len(obs_prob[0])
    numbers = ' '.join(str(elem) for row in obs_prob for elem in row)

    print(rows, cols, numbers)

if __name__ == "__main__":
    main()