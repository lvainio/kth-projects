;; Code written by Leo Vainio
(ns clojure-bernoulli.core
  (:gen-class))

;; Returns binomial (n, k)
(defn binom
  [n k]
  (def r (atom 1))
  (def i (atom 1))

  (while (<= @i k)
    (do
      (reset! r (* @r (/ (+ (- n @i) 1) @i)))
      (swap! i inc)))
    
  (deref r)
  )

;; Helper for inner for loop in bernoulli function
(defn bm-calc
  [m b_nums]
  (def b (atom 0.0))

  (doseq [k (range m)]
    (reset! b (double(- @b (* (double (binom (+ m 1.0) k)) (double(nth b_nums k)))))))
  
  (deref b))


;; Calculates and prints the first n Bernoulli numbers
(defn bernoulli
  [n]
  (def b_nums (atom (vector  1.0)))                   
  (def bm (atom  0.0))                              

  (doseq [m (range 1 (+ n 1))]                              
    (reset! bm (bm-calc m @b_nums))
    (reset! bm (/ @bm (+ m 1)))
    (reset! b_nums (conj @b_nums @bm))
    (reset! bm 0.0))

  ; Print the n first bernoulli numbers
  (doseq [i (range (+ n 1))]
    (println "The " i "th bernoulli number is: " (nth @b_nums i))))                              


(bernoulli 20)
