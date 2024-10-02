"Code written by Leo Vainio"

Object subclass: Bernoulli [
    <comment:
        'I can calculate bernoulli numbers'>

    "Create new bernoulli object"
    Bernoulli class >> new [
        | b |
        b := super new.
        ^b
    ]

    "Calculate the first n bernoulli numbers"
    bernoulli: n [
        | b_nums |
        b_nums := Array new: (n+1).
        b_nums at: 1 put: 1.

        2 to: (n+1) do: [:m |
            b_nums at: m put: 0.

            1 to: (m-1) do: [:k |
                b_nums at: m put: ((b_nums at: m) - ((self binom: m k: (k-1)) * (b_nums at: k))).
            ].
            b_nums at: m put: ((b_nums at: m) / (m)).
        ].

        ^b_nums.                                                         
    ]

    "Calculate the binomial n k"
    binom: n k:k [
        | r |
        r := 1.

        1 to: k do: [:i |
            r := (r * ((n - i + 1) / i)).
        ].

        ^r
    ]
]

"Calculate and print the first 20 bernoulli numbers"
n := 20.
b := Bernoulli new.
b_nums := b bernoulli: n.

1 to: (n+1) do: [:i |
    (i - 1) print.
    'th bernoulli number: ' print.
    (b_nums at: i) printNl.
]