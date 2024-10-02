<?php
// Code written by Leo Vainio

// Calculate the binomial (n, k).
function binom(int $n, int $k) {
    $r = 1;
    for ($i = 1; $i <= $k; $i++) {
        $r = $r * ($n - $i + 1) / $i;
    }
    return $r;
}

// Calculate the first n bernoulli numbers.
function bernoulli(int $n) {
    $b_nums = array(1);
    for($m = 1; $m <= $n; $m++) {
        array_push($b_nums, 0);

        for($k = 0; $k < $m; $k++) {
            $b_nums[$m] = $b_nums[$m] - binom($m + 1, $k) * $b_nums[$k];
        }

        $b_nums[$m] = $b_nums[$m] / ($m + 1);
    }

    return $b_nums;
}

// Prints the n first bernoulli numbers.
function print_n_bernoulli($n) {
    $b_nums = bernoulli($n);

    for($i = 0; $i <= $n; $i++) {
        echo "$i";
        echo "th: ";
        echo $b_nums[$i];
        echo "\n";
    }
}

print_n_bernoulli(20)
?>