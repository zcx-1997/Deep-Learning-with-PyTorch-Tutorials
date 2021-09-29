import  numpy as np
from    matplotlib import pyplot as plt
import  torch



def himmelblau(x):
    ''' 4个极值点：(3,2),(-2.80,3.13),(-3.78,-3.28),(3.58,-1.85) '''
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# [1., 0.] --> [3.0, 2.0],
# [-4, 0.] --> [-3.7793102264404297, -3.2831859588623047],
# [4, 0.]  --> [3.584428310394287, -1.8481265306472778],
# [-1, 0.] --> [-2.8051180839538574, 3.131312608718872]
x = torch.tensor([-1, 0.], requires_grad=True)
optimizer = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):

    pred = himmelblau(x)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print ('step {}: x = {}, f(x) = {}'
               .format(step, x.tolist(), pred.item()))
