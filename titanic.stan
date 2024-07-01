// 『ベイズ推定でKaggleのタイタニック問題を解いてみる』

data {
  int<lower=0> N;            // トレーニングデータのサンプルサイズ
  int<lower=0> K;            // デザイン行列の列数
  array[N] int Y;            // 結果変数
  matrix[N, K] X;           // デザイン行列（推定）
  int<lower=0> N_pred;       // テストデータのサンプルサイズ
  matrix[N_pred, K] X_pred; // デザイン行列（予測）
}

parameters {
  vector[K] beta;
}

transformed parameters {
  vector[N] theta = inv_logit(X * beta);
}

model {
  Y ~ bernoulli(theta);
  beta ~ cauchy(0, 50);
}

generated quantities {
  vector[N_pred] theta_pred;
  vector[N_pred] Y_pred;
  for (n in 1:N_pred) {
    theta_pred[n] = inv_logit(X_pred[n,] * beta);
    Y_pred[n] = bernoulli_rng(theta_pred[n]);
  }
}
