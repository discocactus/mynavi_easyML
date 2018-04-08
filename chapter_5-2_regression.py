
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # 学習データの確認

# In[ ]:


# 学習データを読み込む
# x: 広告費用
# y: クリック数
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:,0]
train_y = train[:,1]


# In[ ]:


# プロット
plt.plot(train_x, train_y, 'o')
plt.show()


# # 数式表現のお勉強

# $$
# \begin{align}
# \sum_{k=1}^{\infty} \frac{1}{k^2} = \frac{\pi^2}{6}
# \end{align}
# $$
$$
\begin{align}
\sum_{k=1}^{\infty} \frac{1}{k^2} = \frac{\pi^2}{6}
\end{align}
$$
# $$
# \newcommand{\rot}[1]{\nabla\times #1}
# \newcommand{\pdfrac}[2]{\frac{\partial #1}{\partial #2}}
# \begin{align}
#   \mathbf{D} &= 0 \\\
#   \mathbf{B} &= 0 \\\
#   \rot{\mathbf{E}} &= - \pdfrac{\mathbf{B}}{t} \\\
#   \rot{\mathbf{H}} &= \pdfrac{\mathbf{D}}{t}
# \end{align}
# $$
$$
\newcommand{\rot}[1]{\nabla\times #1}
\newcommand{\pdfrac}[2]{\frac{\partial #1}{\partial #2}}
\begin{align}
  \mathbf{D} &= 0 \\\
  \mathbf{B} &= 0 \\\
  \rot{\mathbf{E}} &= - \pdfrac{\mathbf{B}}{t} \\\
  \rot{\mathbf{H}} &= \pdfrac{\mathbf{D}}{t}
\end{align}
$$
# # 

# __標準化__
# $$
# \begin{align}
# {z^{(i)}} = \frac{x^{(i)}-u}{\sigma}
# \end{align}
# $$

# In[ ]:


# 標準化(z-score正規化)
# 平均を0、分散を1に変換
# 標準偏差に対してどのくらいの値か
# パラメーターの収束が早くなる
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)


# In[ ]:


# 標準化後プロット
plt.plot(train_z, train_y, 'o')
plt.show()


# # 1次関数として実装

# In[ ]:


# 予測関数
# 切片と傾き
def f(x):
    return theta0 + theta1 * x


# In[ ]:


# 目的関数
# 学習データと予測値の誤差の二乗の総和
# 0.5 は学習に用いる式を導出する際に微分計算で出てくる 2 を相殺するために付けられた定数なので、
# なくても最終的な結果は変わらないが、この定数の大小は収束までの計算回数に若干影響する
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


# In[ ]:


# パラメーターを初期化
# 0〜1の一様乱数
theta0 = np.random.rand()
theta1 = np.random.rand()

# 学習率
ETA = 1e-3

# 誤差の差分
diff = 1

# 更新回数
count = 0

# 誤差の初期値
error = E(train_z, train_y)

# 誤差の差分が計算打ち切り値0.01以下になるまでパラメータ更新を繰り返す
while diff > 1e-2:
    # 各パラメーターの更新は同時に行わなければいけないので、更新結果を一時変数に保存
    # 下の順番で計算する場合だと、theta1の計算時には直前に計算したtheta0ではなく、1つ前のtheta0を使う
    # 1つ前のtheta0 - 学習率 * 目的関数をtheta0で偏微分した結果(合成関数を使って求める)の総和
    tmp0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    # 1つ前のtheta1 - 学習率 * 目的関数をtheta1で偏微分した結果(合成関数を使って求める)の総和
    tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    # パラメータを更新
    theta0 = tmp0
    theta1 = tmp1
    # 前回の誤差との差分を計算
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    # ログの出力
    count += 1
    log = '{}回目: theta0 = {:.3f}, theta1 = {:.3f}, 差分 = {:.4f}'
    print(log.format(count, theta0, theta1, diff))


# In[ ]:


# 結果をプロット
x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()


# # 多項式回帰の実装

# In[ ]:


# パラメータと変数をそれぞれベクトルとみなすことで計算式を簡素化できる


# In[ ]:


# 学習データの行列を作る関数
def to_matrix(x):
    # theta0に対応するx0を1とした行列を転置
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T


# In[ ]:


# 学習データの行列を作る
X = to_matrix(train_z)


# In[ ]:


X


# In[ ]:


# 予測関数
def f(x):
    # 学習データの行列xとパラメータのベクトルthetaの積を求める
    return np.dot(x, theta)


# In[ ]:


np.dot(f(X) - train_y, X)


# In[ ]:


f(X) - train_y


# In[ ]:


# パラメータを初期化
theta = np.random.rand(3)

# 誤差の差分
diff = 1

# 更新回数
count = 0

# 学習を繰り返す
error = E(X, train_y)
while diff > 1e-2:
    # パラメータを更新
    # パラメータのベクトル - 学習率 * 誤差ベクトルと学習データ行列の積
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # 前回の誤差との差分を計算
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error
    # ログの出力
    count += 1
    log = '{}回目: theta = {}, 差分 = {:.4f}'
    print(log.format(count, theta, diff))


# In[ ]:


# 結果をプロット
x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()


# In[ ]:


# 平均二乗誤差
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)


# In[ ]:


# 目的関数(計算打ち切りの条件)を平均二乗誤差で計算

# パラメータを初期化
theta = np.random.rand(3)

# 平均二乗誤差の履歴
errors = []

# 誤差の差分
diff = 1

# 更新回数
count = 0

# 学習を繰り返す
errors.append(MSE(X, train_y))
while diff > 1e-2:
    # パラメータを更新
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    # 前回の誤差との差分を計算
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]
    # ログの出力
    count += 1
    log = '{}回目: theta = {}, 差分 = {:.4f}'
    print(log.format(count, theta, diff))


# In[ ]:


# 結果をプロット
x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()


# In[ ]:


# 誤差をプロット
x = np.arange(len(errors))

plt.plot(x, errors)
plt.show()


# # 確率的勾配降下法の実装

# In[ ]:


# パラメータを初期化
theta = np.random.rand(3)

# 平均二乗誤差の履歴
errors = []

# 誤差の差分
diff = 1

# 更新回数
count = 0

# 学習を繰り返す
errors.append(MSE(X, train_y))
while diff > 1e-2:
    # 学習データを並べ替えるためにランダムな順列を用意する
    p = np.random.permutation(X.shape[0])
    #　学習データをランダムに取り出して確率的勾配降下法でパラメータ更新
    for x, y in zip(X[p,:], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x
    # 前回の誤差との差分を計算
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]
    # ログの出力
    count += 1
    log = '{}回目: theta = {}, 差分 = {:.4f}'
    print(log.format(count, theta, diff))


# In[ ]:


# 結果をプロット
x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()


# In[ ]:


# 誤差をプロット
x = np.arange(len(errors))

plt.plot(x, errors)
plt.show()

