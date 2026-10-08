# import numpy as np
# import matplotlib.pyplot as plt
# # -------------------------
# # contour plot
# # -------------------------
# xvals = np.linspace(0, 10, 500)
# yvals = np.linspace(0, 10, 500)
# X, Y = np.meshgrid(xvals, yvals)
# Z = X**2 + Y

# plt.figure(figsize=(8, 6))

# # choose contour levels up to 36, including the optimal objective value
# levels = [1, 2, 3, 31/9, 5, 8, 12, 16, 20, 24, 28, 32, 36]

# cs = plt.contour(
#     X, Y, Z,
#     levels=levels,
#     colors='gray',
#     linestyles='dotted',
#     linewidths=0.7
# )
# plt.clabel(cs, inline=True, fontsize=8)

# # constraint boundaries
# y1 = (8 - 3*xvals)/2
# y2 = (6 - xvals)/2

# plt.plot(xvals, y1, label='3x + 2y = 8')
# plt.plot(xvals, y2, label='x + 2y = 6')

# # optimal solution
# plt.plot(xvals[X.X], yvals[Y.X], 'ro', markersize=8, label='Optimal solution')
# # plt.plot(x.X, y.X, 'ro', markersize=8, label='Optimal solution')

# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Contours of x^2 + y")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# contour plot
# -------------------------
xvals = np.linspace(0, 10, 500)
yvals = np.linspace(0, 10, 500)
X, Y = np.meshgrid(xvals, yvals)
Z = X**2 + Y

plt.figure(figsize=(8, 6))

# choose contour levels up to 36, including the optimal objective value
levels = [1, 2, 3, 31/9, 5, 8, 12, 16, 20, 24, 28, 32, 36]

cs = plt.contour(
    X, Y, Z,
    levels=levels,
    colors='gray',
    linestyles='dotted',
    linewidths=0.7
)
plt.clabel(cs, inline=True, fontsize=8)

# constraint boundaries
y1 = (8 - 3*xvals)/2
y2 = (6 - xvals)/2

plt.plot(xvals, y1, label='3x + 2y = 8')
plt.plot(xvals, y2, label='x + 2y = 6')

# optimal solution (intersection of the two constraints)
x_opt, y_opt = 1.0, 2.5
plt.plot(x_opt, y_opt, 'ro', markersize=8, label='Optimal solution')

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contours of x^2 + y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()