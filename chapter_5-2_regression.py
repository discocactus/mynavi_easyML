
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


# __標準化(z-score正規化)__  
# 平均を0、分散を1に変換(標準偏差に対してどのくらいの値か)  
# パラメーターの収束が早くなる  
# __$\mu$ (mu) = 平均__  
# __$\sigma$ (sigma) = 分散__    
# $$
# \begin{align}
# {z^{(i)}} = \frac{x^{(i)}-\mu}{\sigma}
# \end{align}
# $$

# In[ ]:


# 標準化(z-score正規化)
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

# __予測関数__  
# 切片と傾き  
# $$
# \begin{align}
# f_\theta = \theta_0 + \theta_1x
# \end{align}
# $$

# In[ ]:


# 予測関数
def f(x):
    return theta0 + theta1 * x


# __目的関数__  
# 学習データと予測値の誤差の二乗の総和  
# $E(\theta)$ が最小となる $\theta$ を求めたい  
# $\frac{1}{2}$ は学習に用いる式を導出する際に微分計算で出てくる 2 を相殺するために付けられた定数なので、  
# なくても最終的な結果は変わらないが、この定数の大小は収束までの計算回数に若干影響する  
# $$
# \begin{align}
# E(\theta) = \frac{1}{2} \sum_{i=1}^{n} \left( y^{(i)} - f_\theta (x^{(i)}) \right) ^2
# \end{align}
# $$

# In[ ]:


# 目的関数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


# __パラメーターの更新式の導出__  
# __最急降下法、勾配降下法__  
# 導関数の符号と逆方向に $\theta$ をずらして(減算)行けば最小値の方に向かう  
# $f_\theta(x)$ は $\theta_0$ と $\theta_1$ を持つ2変数関数なので偏微分を行う  
# __$\eta$ (eta) = 学習率__  
# $$
# \begin{align*}
# & \theta_0 := \theta_0 - \eta \frac{\partial E}{\partial \theta_0} \\
# & \theta_1 := \theta_1 - \eta \frac{\partial E}{\partial \theta_1}
# \end{align*}
# $$

# ここで $\theta_0$, $\theta_1$ はそれぞれ $E(\theta)$ の中の $f_\theta(x)$ の中にあるので、、、  
# $$
# \begin{align*}
# & u = E(\theta) = \frac{1}{2} \sum_{i=1}^{n} \left( y^{(i)} - f_\theta (x^{(i)}) \right) ^2\\
# & v = f_\theta(x) = \theta_0 + \theta_1x
# \end{align*}
# $$
# として、合成関数の微分を行う    
# $$
# \begin{align*}
# & \frac{\partial u}{\partial \theta_0} = \frac{\partial u}{\partial v}\cdot\frac{\partial v}{\partial \theta_0} \\
# & \frac{\partial u}{\partial \theta_1} = \frac{\partial u}{\partial v}\cdot\frac{\partial v}{\partial \theta_1}
# \end{align*}
# $$

# __$\theta_0$ の更新式の導出__  
# ・$u$ を $v$ で微分  
# $$
# \begin{align*}
# \frac{\partial u}{\partial v} & = \frac{\partial}{\partial v} \left( \frac{1}{2} \sum_{i=1}^{n} \left( y^{(i)} - v \right) ^2 \right) \\
# & = \frac{1}{2} \sum_{i=1}^{n} \left( \frac{\partial}{\partial v} \left( y^{(i)} - v \right) ^2 \right) \\
# & = \frac{1}{2} \sum_{i=1}^{n} \left( \frac{\partial}{\partial v} \left( y^{(i)^2} - 2y^{(i)}v +v^2 \right) \right) \\
# & = \frac{1}{2} \sum_{i=1}^{n} \left( -2y^{(i)} + 2v \right) \\
# & = \sum_{i=1}^{n} \left( v -y^{(i)} \right)
# \end{align*}
# $$
# ・$v$ を $\theta_0$ で微分  
# $$
# \begin{align*}
# \frac{\partial v}{\partial \theta_0} & = \frac{\partial}{\partial \theta_0} (\theta_0 + \theta_1 x) \\
# & = 1
# \end{align*}
# $$
# ・それぞれの結果を掛けて、$v$ を $f_\theta(x)$ に戻す  
# $$
# \begin{align*}
# \frac{\partial u}{\partial \theta_0} & = \frac{\partial u}{\partial v}\cdot\frac{\partial v}{\partial \theta_0} \\
# & = \sum_{i=1}^{n} \left( v -y^{(i)} \right) \cdot 1 \\
# & = \sum_{i=1}^{n} \left( f_\theta(x^{(i)}) -y^{(i)} \right)
# \end{align*}
# $$

# __$\theta_1$ の更新式の導出__  
# ・$u$ を $v$ で微分する部分は $\theta_0$ について求めたものと同じ  
# ・$v$ を $\theta_1$ で微分  
# $$
# \begin{align*}
# \frac{\partial v}{\partial \theta_1} & = \frac{\partial}{\partial \theta_1} (\theta_0 + \theta_1 x) \\
# & = x
# \end{align*}
# $$
# ・それぞれの結果を掛けて、$v$ を $f_\theta(x)$ に戻す  
# $$
# \begin{align*}
# \frac{\partial u}{\partial \theta_1} & = \frac{\partial u}{\partial v}\cdot\frac{\partial v}{\partial \theta_1} \\
# & = \sum_{i=1}^{n} \left( v -y^{(i)} \right) \cdot x^{(i)} \\
# & = \sum_{i=1}^{n} \left( f_\theta(x^{(i)}) -y^{(i)} \right) \cdot x^{(i)}
# \end{align*}
# $$

# __パラメーターの更新式__  
# 1つ前の $\theta_0$ - 学習率 \* 目的関数を $\theta_0$ で偏微分して求めた導関数の総和  
# 1つ前の $\theta_1$ - 学習率 \* 目的関数を $\theta_1$ で偏微分して求めた導関数の総和  
# $$
# \begin{align*}
# & \theta_0 := \theta_0 - \eta \sum_{i=1}^{n} \left( f_\theta(x^{(i)}) - y^{(i)} \right) \\
# & \theta_1 := \theta_1 - \eta \sum_{i=1}^{n} \left( f_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
# \end{align*}
# $$

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

# __多項式回帰の予測関数__  
# $$
# \begin{align}
# f_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2
# \end{align}
# $$

# パラメータと変数をそれぞれベクトルとみなすことで計算式を簡単にする  
#   
# $$
# \boldsymbol{\theta} = \left[
# \begin{array}{c}
#   \theta_0 \\
#   \theta_1 \\
#   \theta_2
# \end{array}
# \right]
# \qquad
# \boldsymbol{x(i)} = \left[
# \begin{array}{c}
#   1 \\
#   x^{(i)} \\
#   x^{(i)^2}
# \end{array}
# \right]
# $$

# 学習データが複数あるので1行を1つの学習データとみなして行列として考える  
# 
# $$
# \boldsymbol{X} = \left[
# \begin{array}{c}
#   \boldsymbol{x}^{(1)^T} \\
#   \boldsymbol{x}^{(2)^T} \\
#   \boldsymbol{x}^{(3)^T} \\
#   \vdots \\
#   \boldsymbol{x}^{(n)^T}
# \end{array}
# \right]
# = \left[
# \begin{array}{ccc}
#   1 & x^{(1)} & x^{(1)^2} \\
#   1 & x^{(2)} & x^{(2)^2} \\
#   1 & x^{(3)} & x^{(3)^2} \\
#   & \vdots & \\
#   1 & x^{(n)} & x^{(n)^2}
# \end{array}
# \right]
# $$

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


# 学習データの行列 $\boldsymbol{X}$ とパラメータのベクトル $\boldsymbol{\theta}$ の積をとる
# 
# $$
# f_\boldsymbol{\theta}(\boldsymbol{x}^{(i)}) = \boldsymbol{X\theta} = \left[
# \begin{array}{ccc}
#   1 & x^{(1)} & x^{(1)^2} \\
#   1 & x^{(2)} & x^{(2)^2} \\
#   1 & x^{(3)} & x^{(3)^2} \\
#   & \vdots & \\
#   1 & x^{(n)} & x^{(n)^2}
# \end{array}
# \right]
# \left[
# \begin{array}{c}
#   \theta_0 \\
#   \theta_1 \\
#   \theta_2
# \end{array}
# \right] = \left[
# \begin{array}{c}
#   \theta_0 + \theta_1 x^{(1)} + \theta_2 x^{(1)^2} \\
#   \theta_0 + \theta_1 x^{(2)} + \theta_2 x^{(2)^2} \\
#   \vdots \\
#   \theta_0 + \theta_1 x^{(n)} + \theta_2 x^{(n)^2}
# \end{array}
# \right]
# $$

# In[ ]:


# 予測関数
def f(x):
    # 学習データの行列xとパラメータのベクトルthetaの積を求める
    return np.dot(x, theta)


# In[ ]:


# 参考
f(X)


# In[ ]:


# 参考
theta


# In[ ]:


# 参考
# f(X)[0] =
theta[0] + theta[1] * X[0][1] + theta[2] * X[0][2]


# __追加された $\theta_2$ の更新式の導出__  
# ・$u$ を $v$ で微分する部分は $\theta_0$ について求めたものと同じ  
# ・$v$ を $\theta_2$ で微分  
# $$
# \begin{align*}
# \frac{\partial v}{\partial \theta_2} & = \frac{\partial}{\partial \theta_2} (\theta_0 + \theta_1 x +\theta_2 x^2) \\
# & = x^2
# \end{align*}
# $$
# ・それぞれの結果を掛けて、$v$ を $f_\theta(x)$ に戻す  
# $$
# \begin{align*}
# \frac{\partial u}{\partial \theta_2} & = \frac{\partial u}{\partial v}\cdot\frac{\partial v}{\partial \theta_2} \\
# & = \sum_{i=1}^{n} \left( v -y^{(i)} \right) \cdot x^{(i)^2} \\
# & = \sum_{i=1}^{n} \left( f_\theta(x^{(i)}) -y^{(i)} \right) \cdot x^{(i)^2}
# \end{align*}
# $$

# __パラメーターの更新式__  
# 1つ前の $\theta_0$ - 学習率 \* 目的関数を $\theta_0$ で偏微分して求めた導関数の総和  
# 1つ前の $\theta_1$ - 学習率 \* 目的関数を $\theta_1$ で偏微分して求めた導関数の総和  
# 1つ前の $\theta_2$ - 学習率 \* 目的関数を $\theta_2$ で偏微分して求めた導関数の総和  
# $$
# \begin{align*}
# & \theta_0 := \theta_0 - \eta \sum_{i=1}^{n} \left( f_\theta(x^{(i)}) - y^{(i)} \right) \\
# & \theta_1 := \theta_1 - \eta \sum_{i=1}^{n} \left( f_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)} \\
# & \theta_2 := \theta_2 - \eta \sum_{i=1}^{n} \left( f_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)^2}
# \end{align*}
# $$
# 
# __パラメーターの数を $j$ として一般化__  
# $$
# \begin{align*}
# & \theta_j := \theta_j - \eta \sum_{i=1}^{n} \left( f_\boldsymbol{\theta}(\boldsymbol{x}^{(i)}) - y^{(i)} \right) x_j^{(i)}
# \end{align*}
# $$

# ここで $j$=0 の時、$f_\theta(\boldsymbol{x}^{(i)}) - y^{(i)}$ と $x_0^{(i)}$ をそれぞれベクトルとして考える
# $$
# \boldsymbol{f} = \left[
# \begin{array}{c}
#   f_\theta(\boldsymbol{x}^{(1)}) - y^{(1)} \\
#   f_\theta(\boldsymbol{x}^{(2)}) - y^{(2)} \\
#   \vdots \\
#   f_\theta(\boldsymbol{x}^{(n)}) - y^{(n)}
# \end{array}
# \right]
# \qquad
# \boldsymbol{x_0} = \left[
# \begin{array}{c}
#   x_0^{(1)} \\
#   x_0^{(2)} \\
#   \vdots \\
#   x_0^{(n)}
# \end{array}
# \right]
# $$
# この $\boldsymbol{f}$ を転置して $\boldsymbol{x_0}$ と掛け合わせれば更新式の総和の部分と同じになる  
# 
# 同様にして $\boldsymbol{f}^{\mathrm{T}}$ と $x_j^{(i)}$ の行列 $\boldsymbol{X}$ を掛け合わせたものを更新式に用いる

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

