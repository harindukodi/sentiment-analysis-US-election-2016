from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

style.use("ggplot")

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
plt.xlabel('my data', fontsize=14, color='red')

plt.legend()
def animate(i):
    pullData = open("twitter-out3.txt", "r").read()
    lines = pullData.split('\n')

    xar = []
    yar = []

    x =0
    y=0

    for l in lines:
        x+=1
        if '1' in l:
            y+=1
            # y=1
        elif '0' in l:
            y-=1
            # y=0
        xar.append(x)
        yar.append(y)

    ax1.clear()
    ax1.plot(xar, yar)
    # ax1.plot(xar, "g", label='ll')
    # ax1.plot(yar, label='ff')
    sew='Live Sentiment Analysis - Hillary Clinton \n \n X axis - Tweet Count \n Y axis - Sentiment Score'
    ax1.set_title(sew)

ani = animation.FuncAnimation(fig, animate, interval=1000)

plt.show()

