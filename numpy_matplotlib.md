---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# NumPy, Matplotlibの使い方サンプル


NumPyの`import`

```python
import numpy as np
```

Matplotlib の`import`

```python
import matplotlib.pyplot as plt
```

## NumPy


2次元配列

```python
np.array([[1, 2, 3], [4, 5, 6]])
```

```python
np.array(np.mat("1 2 3 ; 4 5 6"))
```

3$\times$4一様乱数行列

```python
np.random.rand(3, 4)
```

`np.arange()`関数

```python
np.arange(2, -3, -1)
```

```python
np.arange(1, 7)
```

```python
np.arange(8)
```

要素へのアクセス

```python
a = np.arange(8)
a
```

```python
a[0:4]
```

```python
a[0:4:2]
```

```python
a[4:]
```

```python
a[0::2]
```

```python
a[0::2] = 10
a
```

要素へのアクセス(2次元)

```python
a = np.array(np.mat("1 2 3 ; 4 5 6"))
a
```

```python
a[0]
```

```python
a[0, 2]
```

```python
a[:, 2]
```

```python
a[:, ::2]
```

```python
a[:, ::2] *= 10
a
```

要素の並び替えおよび1次元化

```python
a = np.array(np.mat("1 2 3 ; 4 5 6"))
a
```

```python
a.shape
```

```python
a.reshape((3, 2))
```

```python
a.reshape((3, 2), order='F')
```

```python
a.flatten()
```

```python
a.flatten('F')
```

要素の連結

```python
a = np.arange(0, 4)
b = np.arange(4, 8)
np.r_[a, b]
```

```python
np.c_[a, b]
```

```python
np.hstack([a, b])
```

```python
np.vstack([a, b])
```

行列と行列の加算

```python
a = np.arange(4).reshape((2, 2))
b = np.arange(4, 8).reshape((2, 2))
a, b
```

```python
a+b
```

行列とスカラの演算

```python
a*2
```

行列積

```python
a.dot(b)
```

行列の要素毎の乗算

```python
a*b
```

正弦値

```python
x1 = np.arange(5)*np.pi/4
x2 = x1*2
x = np.vstack([x1, x2])
x
```

```python
np.sin(x)
```

## Matplotlib


プロット例

```python
x1 = np.arange(25)*np.pi/12
x1
```

```python
y1 = np.sin(x1)
y2 = np.cos(x1)
plt.plot(x1, y1, x1, y2)
plt.rcParams.update({"font.size": 14})
plt.tight_layout()
plt.savefig("plot.png")
plt.savefig("plot.pdf")
plt.show()
plt.close()
```

プロット例(分割してプロット)

```python
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
```
