// Code written by Leo Vainio.

use std::io;

fn main() {
    println!("Enter a positive integer number: ");

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
        
    match input.trim().parse::<i32>() {
        Ok(n) => println!("The {}th Bernoulliou number is: {}", n, bernoullio(n)),
        Err(_) => println!("Invalid input"),
    }
}

// Get the nth Bernoulliou number.
fn bernoullio(n: i32) -> f64 {
    let nth = (n+1) as usize;
    let mut b_nums: Vec<f64> = vec![0.0; nth];

    b_nums[0] = 1.0; 

    for m in 1..nth {
        // println!("{}", m);
        b_nums[m] = 0.0;
        for k in 0..m {
            b_nums[m] = b_nums[m] - binom(m + 1, k) * b_nums[k];
        }
        b_nums[m] = b_nums[m] / (m+1) as f64;
    }
    return b_nums[n as usize];
}

// Calculate binomials (n, k).
fn binom(n: usize, k: usize) -> f64 {
    let mut r = 1.0;
    
    
    for i in 1..(k+1) {
        
        r = r * (n - i + 1) as f64/i as f64;
    }

    return r;
}

