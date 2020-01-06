import os
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl



set3 = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors

def plot_distribution(y1, y2, ylims=[500, 500]):
    set3 = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors

    data_names=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(np.arange(7),y1, 0.5, color=set3, alpha=0.8)

    ax1.set_xticklabels(data_names, rotation=60, fontsize=6)
    ax1.set_xlim([0, 7])
    ax1.set_ylim([0, 2500])
    ax1.set_title("Train Data")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(np.arange(7),y2, 0.5, color=set3, alpha=0.8)

    ax2.set_xticklabels(data_names, rotation=60, fontsize=6)
    ax2.set_xlim([0, 7])
    ax2.set_ylim([0, 300])
    ax2.set_title("Test Data")
    plt.tight_layout()
    plt.show()

#os.chdir("..\..")
curpath = os.getcwd()
readpath=curpath+"\\last_traintest"
tr = []
ts = []
print(readpath)
for root, dirs, files in os.walk(readpath):
    path_list = root.split(os.sep)
    print(root)

    if(path_list[-2])=='train':
        if (len(files) > 0):
            if(path_list[-1])=='0':
                tr.append(len(files))
            if (path_list[-1]) == '1':
                tr.append(len(files))
            if (path_list[-1]) == '2':
                tr.append(len(files))
            if (path_list[-1]) == '3':
                tr.append(len(files))
            if (path_list[-1]) == '4':
                tr.append(len(files))
            if (path_list[-1]) == '5':
                tr.append(len(files))
            if (path_list[-1]) == '6':
                tr.append(len(files))
    if (path_list[-2]) == 'test':
        if (len(files) > 0):
            if (path_list[-1]) == '0':
                ts.append(len(files))
            if (path_list[-1]) == '1':
                ts.append(len(files))
            if (path_list[-1]) == '2':
                ts.append(len(files))
            if (path_list[-1]) == '3':
                ts.append(len(files))
            if (path_list[-1]) == '4':
                ts.append(len(files))
            if (path_list[-1]) == '5':
                ts.append(len(files))
            if (path_list[-1]) == '6':
                ts.append(len(files))
print(tr)
print(ts)
plot_distribution(tr, ts, ylims=[8000, 1000])
