import matplotlib.pyplot as plt

from tnnpy import GodelEncoder
from tnnpy.plotting import plot_cylinders
from tnnpy.utils import get_invariant_partition_2d

alpha_symbols = ["a", "b", "c", "d"]
beta_symbols = ["1", "2", "3", "4"]
alpha_seq = ["b", "a", "c"]
beta_seq = ["1", "3", "3", "4"]

cylinders = get_invariant_partition_2d(alpha_symbols, beta_symbols, alpha_seq, beta_seq)
x_ticks = [GodelEncoder(alpha_symbols).encode_sequence(s) for s in alpha_symbols] + [1.0]
y_ticks = [GodelEncoder(beta_symbols).encode_sequence(s) for s in beta_symbols] + [1.0]

fig = plt.figure()
plt.style.use("ggplot")
ax = fig.add_subplot(111, aspect="equal")
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
plot_cylinders(ax, cylinders)
plt.show()
