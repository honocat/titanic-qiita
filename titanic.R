## 『ベイズ推定でKaggleのタイタニック問題を解いてみる』

# 準備
## パッケージの読み込み
pacman::p_load(tidyverse,
               cmdstanr,
               posterior)
## 処理の高速化
options(mc.cores = parallel::detectCores())
## データの読み込み
myd_train <- read_csv('train.csv')
myd_test <- read_csv('test.csv')
## myd_testに`Survived`を追加
myd_test$Survived <- NA
## データの結合
myd <- rbind(myd_train, myd_test)
## 分析に用いる変数変数のみ抽出
myd <- myd |>
  select(-c(Name, Ticket, Cabin, Embarked))
## glimpse()
glimpse(myd)

# 1. データ可視化・理解
## 生存・死亡それぞれの男女比
myd |>
  ggplot() +
  geom_bar(aes(x = as.factor(Survived),
               fill = Sex)) +
  labs(x = NULL, y = '人数') +
  scale_fill_discrete(name = '性別',
                      labels = c('女性', '男性')) +
  theme_minimal()
myd |>
  ggplot() +
  geom_histogram(aes(x = Age,
                     y = after_stat(density)),
                 color = 'black') +
  labs(x = '年齢', y = '密度') +
  theme_minimal()
myd |>
  ggplot() +
  geom_histogram(aes(x = Fare,
                     y = after_stat(density),
                     fill = as.factor(Pclass))) +
  labs(x = '料金', y = '密度') +
  scale_fill_discrete(name = 'クラス') +
  theme_minimal()

# 2. データ前処理・加工
## NAを除去したmyd_NAomitを作成
myd_NAomit <- na.omit(myd) |>
  select(Pclass, Age, Fare)
## 平均値の計算
empty_vec <- rep(NA, times = 2)      # 空のベクトルを作成
empty_vec[1] <- mean(myd_NAomit$Age) # Ageの平均
empty_vec[2] <- myd_NAomit |>        # Fareの平均
  filter(Pclass == 3) |>             ## Pclass = 3でフィルタリング
  select(Fare) |>                    ## Fareのみを抽出
  unlist() |>                        ## リストを解除
  mean()                             ## 平均値の計算
## 平均値をもとのデータに代入
myd <- myd |>
  mutate(Age  = ifelse(!is.na(Age),  Age,  empty_vec[1]),
         Fare = ifelse(!is.na(Fare), Fare, empty_vec[2]))
## トレーニングデータとテストデータに分ける
myd_train <- myd |>
  filter(!is.na(Survived) == TRUE)
myd_test <- myd |>
  filter(!is.na(Survived) == FALSE)

# 4. データリストの作成
## デザイン行列（推定）の作成
formula_titanic <- formula(Survived ~ Pclass + Sex + Age
                            + SibSp + Parch + Fare)
design_train <- model.matrix(formula_titanic, data = myd_train)
head(design_train)
## デザイン行列（予測）の作成
design_test <- myd_test[, c(-1, -2)] |>              # PassengerId, Survivedを除去
  mutate(Intercept = rep(1, times = n()),            # 切片を追加
         Sex       = ifelse(Sex == 'male', 1, 0))    # Sexをダミー変数に変換
design_test <- design_test[, c(7, 1, 2, 3, 4, 5, 6)] # 変数を並び替え．
head(design_test)
## データをリストにまとめる
myd_list <- list(
  N = nrow(myd_train),     # トレーニングデータのサンプルサイズ
  K = 7,                   # デザイン行列の列数
  Y = myd_train$Survived,  # 結果変数
  X = design_train,        # デザイン行列（推定）
  N_pred = nrow(myd_test), # テストデータのサンプルサイズ
  X_pred = design_test     # デザイン行列（予測）
)

# 6. MCMCの実行
## Stanファイルのコンパイル
stan_titanic <- cmdstan_model('titanic.stan')
## MCMC!
fit_titanic <- stan_titanic$sample(
  data = myd_list,     # 引き渡すデータ
  seed = 1912-04-14,   # 乱数の種
  chains = 4,          # チェイン数
  refresh = 1000,      # コンソールに表示する結果の間隔
  iter_warmup = 1000,  # バーンイン期間
  iter_sampling = 3000 # サンプリング数
)

# 7. 結果の確認
## rhat < 1.1
all(fit_titanic$summary()[, 'rhat'] < 1.1)
## トレースプロット
### 結果をデータフレームとして取得
post_titanic <- fit_titanic$draws() |>
  as_draws_df() |>
  mutate(chains = as.factor(.chain))
### trace plot
post_titanic |>
  ggplot() +
  geom_line(aes(x = .iteration,
                y = `beta[1]`,
                color = chains)) +
  labs(x = 'iteration', y = expression(beta[1])) +
  theme_minimal()

# 結果の提出
## 予測値の確認
fit_titanic$summary('Y_pred')
## 生存確率をベクトルとして取得
res <- fit_titanic$summary('Y_pred')[, 'mean'] |> unlist()
## 各乗客ごとに生存か死亡を判断
for (n in 1:nrow(myd_test)) {
  myd_test$Survived[n] <- ifelse(res[n] < 0.5, 0, 1)
}
myd_test$Survived |> head(10)
## 可視化
myd_test |>
  ggplot() +
  geom_bar(aes(x = as.factor(Survived),
               fill = Sex)) +
  labs(x = NULL, y = '人数') +
  scale_fill_discrete(name = '性別',
                      labels = c('女性', '男性')) +
  theme_minimal()
