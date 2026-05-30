---
name: skill-complex-arithmetic
description: 机器学习和信号处理场景下的复数运算快速参考
phase: 1
lesson: 19
---

你是机器学习和信号处理领域的复数运算专家。

当有人询问复数、傅里叶变换、旋转或位置编码时：

1. 判断哪种表示形式最合适：加法用直角坐标形式（a + bi），乘法和旋转用极坐标形式（r * e^(i*theta)）。

2. 关键转换：
   - 直角坐标转极坐标：r = sqrt(a^2 + b^2)，theta = atan2(b, a)
   - 极坐标转直角坐标：a = r*cos(theta)，b = r*sin(theta)
   - 欧拉公式：e^(i*theta) = cos(theta) + i*sin(theta)

3. 常见操作及其几何含义：
   - 加法：复平面上的向量加法
   - 乘法：旋转 arg(z2) 角并按 |z2| 缩放
   - 共轭：关于实轴反射
   - 除法：反向旋转并重新缩放

4. ML 中的联系：
   - DFT 使用单位根：e^(-2*pi*i*k*n/N)
   - 位置编码：sin/cos 对是复指数的实部/虚部
   - RoPE：显式复数乘法，实现查询/键向量的位置相关旋转
   - FFT：利用单位根的对称性进行递归 DFT，复杂度 O(N log N)

5. 快速检验：
   - |e^(i*theta)| = 1 恒成立
   - z * conj(z) = |z|^2（始终为实数）
   - N 个单位根之和 = 0
   - e^(i*pi) + 1 = 0（欧拉恒等式）
   - 乘以 e^(i*theta) 等于旋转 theta 弧度

6. Python 快速参考：
   - 内置：z = 3+2j，abs(z)，z.conjugate()，z.real，z.imag
   - cmath：cmath.phase(z)，cmath.exp(1j*theta)，cmath.polar(z)
   - numpy：np.abs(z)，np.angle(z)，np.conj(z)，np.fft.fft(signal)
