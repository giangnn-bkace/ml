# Imports
import numpy as np
import matplotlib.pyplot as plt

# Create a new figure of size 8x6 points, using 100 dots per inch
plt.figure(figsize=(10,6), dpi=100)

# Create a new subplot from a grid of 1x1
plt.subplot(111)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

# Plot cosine using blue color with a continuous line of width 1 (pixels)
plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label='cosine')

# Plot sine using green color with a continuous line of width 1 (pixels)
plt.plot(X, S, color="red", linewidth=2.5, linestyle="-", label='sin')

# Set x limits
plt.xlim(X.min()*1.1, X.max()*1.1)

# Set x ticks
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
			[r'$-\pi$',r'$-\pi/2$',r'$0$',r'$\pi/2$',r'$\pi$'])

# Set y limits
plt.ylim(C.min()*1.1, C.max()*1.1)

# Set y ticks
plt.yticks([-1, 0, 1], [r'$-1$',r'$0$',r'$+1$'])

# Move spines
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# Adding legend
plt.legend(loc='upper left', frameon=True)

# Save figure using 72 dots per inch
#plt.savefig("./figures/helloMatplotlib.png",dpi=600)

# Show result on screen
plt.show()