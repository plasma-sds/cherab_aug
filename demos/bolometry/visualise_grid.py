
import matplotlib.pyplot as plt
from cherab.aug.machine.wall_outline import plot_aug_wall_outline
from cherab.aug.bolometry import load_standard_inversion_grid


grid = load_standard_inversion_grid()

plot_aug_wall_outline()

plt.ion()
plot_aug_wall_outline()
for i in range(grid.count):
    p1, p2, p3, p4 = grid[i]
    plt.plot([p1.x, p2.x], [p1.y, p2.y], 'g')
    plt.plot([p2.x, p3.x], [p2.y, p3.y], 'g')
    plt.plot([p3.x, p4.x], [p3.y, p4.y], 'g')
    plt.plot([p4.x, p1.x], [p4.y, p1.y], 'g')

plt.ioff()
plt.show()
