
EVD = []
y = []
i = 2
while i < 11:
    maxiter = 10
    tn = RF * i
    y.append(int(tn))
    traj_set, rewards_gt = trajectory_set(F, RF, tn, tl)
    maxC = dpmhl(traj_set, maxiter, tn)
    e = evd(maxC, rewards_gt, maxiter, tn)
    EVD.append(e)
    print("EVD = ", EVD)
    i = i + 2
print("Completed. EVD = ", EVD)
plt.plot(y, np.asarray(EVD))
plt.xlabel('no. of trajectories per agent')
plt.ylabel('EVD for the new trajectory')
plt.savefig('figure.png')
plt.show()



#    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 6))
#    a = r.reshape((M, N))
#    z1 = ax1.imshow(a)
#    plt.colorbar(z1, ax=ax1, fraction=0.046, pad=0.2)
#    plt.show()

    size = 8
    data = np.arange(size * size).reshape((size, size))

    # Limits for the extent
    x_start = 0.0
    x_end = 8.0
    y_start = 8.0
    y_end = 0.0

    extent = [x_start, x_end, y_start, y_end]

    # The normal figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(reward_feature(M, N, r[0]), extent=extent, origin='lower', interpolation='None', cmap='viridis')

    # Add the text
    jump_x = (x_end - x_start) / (2.0 * size)
    jump_y = (y_end - y_start) / (2.0 * size)
    x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
    y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = data[y_index, x_index]
            text_x =  x + jump_x
            text_y = y +jump_y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center')

    fig.colorbar(im)
    plt.show()