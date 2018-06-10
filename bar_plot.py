import numpy as np
import matplotlib.pyplot as plt

N = 3
ind = np.arange(N)  # the x locations for the groups
width = 0.27       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

yvals = [.441, .449, .436]
rects1 = ax.bar(ind, yvals, width, color='cadetblue')
zvals = [.928, .936, .921]
rects2 = ax.bar(ind+width, zvals, width, color='#359920')
kvals = [.985, .985, .984]
rects3 = ax.bar(ind+width*2, kvals, width, color='#c12a0b')

ax.set_ylabel('Scores', fontsize=16)
ax.set_xticks(ind+width)
ax.set_xticklabels( ('F1', 'Precision', 'Recall'), fontsize=16 )
ax.legend( (rects1[0], rects2[0], rects3[0]), ('Random Forest', 'AdaBoost', 'GradientBoosting'), loc='lower right')

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., .995*h, h,
                ha='center', va='bottom', fontsize=14)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()
