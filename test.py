import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

refCC = (-11,-4,37)
mouCC = (2,-1,40)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)


if plt.fignum_exists(1):
    x1 = 1 * np.outer(np.cos(u), np.sin(v))
    y1 = 1 * np.outer(np.sin(u), np.sin(v))
    z1 = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x1, y1, z1,  rstride=4, cstride=4, color='k', linewidth=0, alpha=0.5)

    x = 1 * np.outer(np.cos(u), np.sin(v)) + refCC[0]
    y = 1 * np.outer(np.sin(u), np.sin(v)) + refCC[2]               # Since we're plotting the y on the z-axis and vice versa
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + refCC[1]     # Since we're plotting the y on the z-axis and vice versa

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    mx = 1 * np.outer(np.cos(u), np.sin(v)) + mouCC[0]
    my = 1 * np.outer(np.sin(u), np.sin(v)) + mouCC[2]
    mz = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + mouCC[1]

    ax.plot_surface(mx, my, mz,  rstride=4, cstride=4, color='r', linewidth=0, alpha=0.5)

    plt.draw()
    print("Drawing new frame")
else:
    fig = plt.figure(figsize=(12,12), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    #ax.set_aspect('equal')

    x1 = 1 * np.outer(np.cos(u), np.sin(v))
    y1 = 1 * np.outer(np.sin(u), np.sin(v))
    z1 = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x1, y1, z1,  rstride=4, cstride=4, color='k', linewidth=0, alpha=0.5)
    
    x = 1 * np.outer(np.cos(u), np.sin(v)) + refCC[0]
    y = 1 * np.outer(np.sin(u), np.sin(v)) + refCC[2]               # Since we're plotting the y on the z-axis and vice versa
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + refCC[1]     # Since we're plotting the y on the z-axis and vice versa

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    mx = 1 * np.outer(np.cos(u), np.sin(v)) + mouCC[0]
    my = 1 * np.outer(np.sin(u), np.sin(v)) + mouCC[2]
    mz = 1 * np.outer(np.ones(np.size(u)), np.cos(v)) + mouCC[1]

    ax.plot_surface(mx, my, mz,  rstride=4, cstride=4, color='r', linewidth=0, alpha=0.5)

    ### ------------- ###
#        figure = plt.figure(1)
#        ax = figure.add_subplot(111, projection='3d')
    
    ### Point option    ###
#        ax.scatter(refCC[0], refCC[1], refCC[2], c='blue', marker='2')
#        ax.scatter(mouCC[0], mouCC[1], mouCC[2], c='red', marker='*')

    ax.set_xlabel('X Label')    
    ax.set_ylabel('Z Label (for viewing only)')
    ax.set_zlabel('Y Label (for viewing only)')

    ### Set the plot angle and view     ###
    ax.view_init(elev=10., azim=-80)

    plt.xlim((-1.2*refCC[0],refCC[0]*1.2))
    plt.ylim((0,refCC[2]*1.2))
    plt.gca().invert_xaxis()
    plt.show()

