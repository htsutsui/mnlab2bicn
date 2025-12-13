# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NumPy, Matplotlibの使い方サンプル

# %% [markdown]
# NumPyの`import`

# %%
import numpy as np

# %% [markdown]
# Matplotlib の`import`

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ## NumPy

# %% [markdown]
# 2次元配列

# %%
np.array([[1, 2, 3], [4, 5, 6]])

# %%
np.array(np.mat("1 2 3 ; 4 5 6"))

# %% [markdown]
# 3$\times$4一様乱数行列

# %%
np.random.rand(3, 4)

# %% [markdown]
# `np.arange()`関数

# %%
np.arange(2, -3, -1)

# %%
np.arange(1, 7)

# %%
np.arange(8)

# %% [markdown]
# 要素へのアクセス

# %%
a = np.arange(8)
a

# %%
a[0:4]

# %%
a[0:4:2]

# %%
a[4:]

# %%
a[0::2]

# %%
a[0::2] = 10
a

# %% [markdown]
# 要素へのアクセス(2次元)

# %%
a = np.array(np.mat("1 2 3 ; 4 5 6"))
a

# %%
a[0]

# %%
a[0, 2]

# %%
a[:, 2]

# %%
a[:, ::2]

# %%
a[:, ::2] *= 10
a

# %% [markdown]
# 要素の並び替えおよび1次元化

# %%
a = np.array(np.mat("1 2 3 ; 4 5 6"))
a

# %%
a.shape

# %%
a.reshape((3, 2))

# %%
a.reshape((3, 2), order='F')

# %%
a.flatten()

# %%
a.flatten('F')

# %% [markdown]
# 要素の連結

# %%
a = np.arange(0, 4)
b = np.arange(4, 8)
np.r_[a, b]

# %%
np.c_[a, b]

# %%
np.hstack([a, b])

# %%
np.vstack([a, b])

# %% [markdown]
# 行列と行列の加算

# %%
a = np.arange(4).reshape((2, 2))
b = np.arange(4, 8).reshape((2, 2))
a, b

# %%
a+b

# %% [markdown]
# 行列とスカラの演算

# %%
a*2

# %% [markdown]
# 行列積

# %%
a.dot(b)

# %% [markdown]
# 行列の要素毎の乗算

# %%
a*b

# %% [markdown]
# 正弦値

# %%
x1 = np.arange(5)*np.pi/4
x2 = x1*2
x = np.vstack([x1, x2])
x

# %%
np.sin(x)

# %% [markdown]
# ## Matplotlib

# %% [markdown]
# プロット例

# %%
x1 = np.arange(25)*np.pi/12
x1

# %%
y1 = np.sin(x1)
y2 = np.cos(x1)
plt.plot(x1, y1, x1, y2)
plt.rcParams.update({"font.size": 14})
plt.tight_layout()
plt.savefig("plot.png")
plt.savefig("plot.pdf")
plt.show()
plt.close()

# %% [markdown]
# プロット例(分割してプロット)

# %%
plt.subplot(2, 1, 1)
plt.plot(x1, y1, "r-o", label="sin")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.title("Example of subplot")
plt.legend()
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(x1, y2, "k.", label="cos")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.rcParams.update({"font.size": 14})
plt.tight_layout()
plt.savefig("plot.png")
plt.savefig("plot.pdf")
plt.show()
plt.close()
