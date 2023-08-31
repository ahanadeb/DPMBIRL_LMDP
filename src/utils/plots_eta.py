from matplotlib import pyplot as pl
import numpy as np

pl.clf()

a = np.asarray([.8,.68,.57,.536,.533,.43])
err_a1 = np.asarray([.15,.1,.2,.1,.15,.05])
err_a2 = np.asarray([.1,.1,.15,.2,.1,.15])

b=np.asarray([.86,.87,.84,.79,.74,.62])
err_b1=np.asarray([.3,.2,.2,.1,.3,.2])/2
err_b2=np.asarray([.15,.25,.1,.2,.15,.1])/2

c=np.asarray([.8,.76,.5,.5,.45,.42])
err_c1=np.asarray([.1,.15,.3,.2,.1,.15])/3
err_c2=np.asarray([.25,.15,.25,.1,.15,.2])/3





# x = np.linspace(0, 30, 30)
x=np.arange(len(b))
x=x*2 +1
# y = np.sin(x/6*np.pi)

ax = pl.gca()
# b += np.random.normal(0, 0.1, size=b.shape)
ax.plot(x,b, 'k', color='#ffc671')
ax.fill_between(x,b-err_b1, b+err_b2,
    alpha=0.5, edgecolor='#ffc671', facecolor='#ffe371',label='_nolegend_')

ax.plot(x,a, 'k', color='#1B2ACC')
ax.fill_between(x,a-err_b1, a+err_b2,
    alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF',label='_nolegend_')

ax.plot(x,c, 'k', color='#ff5c8f')
ax.fill_between(x,c-err_c1, c+err_c2,
    alpha=0.5,  edgecolor='#ff5c8f', facecolor='#ff9bbb',label='_nolegend_')


#plt.legend(["DPMH(Langevin)","EM(3)","EM(6)","EM(9)"])
ax.legend(["$\eta=5$","$\eta=10$","$\eta=30$"])
pl.ylabel("average EVD")
pl.xlabel("no of trajectories per agent")
# y = np.cos(x/6*np.pi)
# error = np.random.rand(len(y)) * 0.5
# y += np.random.normal(0, 0.1, size=y.shape)
# pl.plot(x, y, 'k', color='#1B2ACC')
# pl.fill_between(x, y-error, y+error,
#     alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
#     linewidth=4, linestyle='dashdot', antialiased=True)



# y = np.cos(x/6*np.pi)  + np.sin(x/3*np.pi)
# error = np.random.rand(len(y)) * 0.5
# y += np.random.normal(0, 0.1, size=y.shape)
# pl.plot(x, y, 'k', color='#3F7F4C')
# pl.fill_between(x, y-error, y+error,
#     alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
#     linewidth=0)



pl.show()