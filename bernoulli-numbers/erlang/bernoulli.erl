%% Code written by Leo Vainio.

-module(bernoulli).

%% The functions that can be called from this module.
-export([bernoulli/1]).


% Calculates and prints the n first bernoulli numbers.
bernoulli(N) -> 
    B = [1],
    M = 1,

    B_nums = b_outer_loop(B, M, N),

    print_result(B_nums, 0).

% Prints out all elements of a list.
print_result([], _) -> 
    ok;
print_result([H|T], N) ->
    io:fwrite("~pth bernoulli nr: ~p\n", [N, H]),
    print_result(T, N+1).

% Calculates the outer loop in the bernoulli function.
b_outer_loop(B, M, N) when M > N -> B;
b_outer_loop(B, M, N) when M =< N ->
    K = 0,
    Bm = b_inner_loop(B, K, M, 0),
    B_new = B ++ [Bm / (M+1)],
    b_outer_loop(B_new, M+1, N).

% Calculates the inner loop in the bernoulli function.
b_inner_loop(_, K, M, Bm) when K == M -> Bm;
b_inner_loop(B, K, M, Bm) when K < M ->
    Bm_new = Bm - binom(M+1, K) * lists:nth(K+1, B),
    b_inner_loop(B, K+1, M, Bm_new).

% Calculates the binomial (n, k).
binom(N, K) ->
    R = 1,
    I = 1,
    binom_loop(R, N, K, I).

% Binomial "for loop".
binom_loop(R, _, K, I) when I > K -> R;
binom_loop(R, N, K, I) when I =< K ->
    R_new = R * (N - I + 1) / I,
    binom_loop(R_new, N, K, I + 1).