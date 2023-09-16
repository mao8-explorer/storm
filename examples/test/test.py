import matplotlib.pyplot as plt

fig, ax = plt.subplots()
text = ax.text(0.5, 0.5, 'event', ha='center', va='center', fontdict={'size': 20})


def call_back(event):
    info = 'name:{}\n button:{}\n x,y:{},{}\n xdata,ydata:{}{}'.format(event.name, event.button,event.x, event.y,event.xdata, event.ydata)
    text.set_text(info)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect('button_press_event', call_back)
# fig.canvas.mpl_connect('button_release_event', call_back)
# fig.canvas.mpl_connect('motion_notify_event', call_back)

plt.show()