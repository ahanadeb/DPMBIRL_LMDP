import matplotlib.pyplot as plt
import numpy as np


def plot_3(A, B, C, a, b, M, N):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    a = a.reshape((M, N))
    b = b.reshape((M, N))
    z1 = ax1.imshow(a)
    z2 = ax2.imshow(b)
    # z3 = ax3.imshow(np.abs(a - b))
    z3 = ax3.imshow((a - b))
    ax1.title.set_text(A)
    ax2.title.set_text(B)
    ax3.title.set_text(C)
    plt.colorbar(z1, ax=ax1, fraction=0.046, pad=0.2)
    plt.colorbar(z2, ax=ax2, fraction=0.046, pad=0.2)
    plt.colorbar(z3, ax=ax3, fraction=0.046, pad=0.2)
    plt.show()
    return plt


def plot_4(A, B, C, D, a, b, c, d):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
    a = a.reshape((10, 10))
    b = b.reshape((10, 10))
    c = c.reshape((10, 10))
    d = d.reshape((10, 10))
    z1 = ax1.imshow(a, vmax=0, vmin=-8.5)
    z2 = ax2.imshow(b, vmax=0, vmin=-8.5)
    z3 = ax3.imshow(c, vmax=0, vmin=-8.5)
    z4 = ax4.imshow(d, vmax=0, vmin=-8.5)
    ax1.title.set_text(A)
    ax2.title.set_text(B)
    ax3.title.set_text(C)
    ax4.title.set_text(D)
    plt.colorbar(z1, ax=ax1, fraction=0.046, pad=0.2)
    plt.colorbar(z2, ax=ax2, fraction=0.046, pad=0.2)
    plt.colorbar(z3, ax=ax3, fraction=0.046, pad=0.2)
    plt.colorbar(z4, ax=ax4, fraction=0.046, pad=0.2)
    plt.show()
    return plt
