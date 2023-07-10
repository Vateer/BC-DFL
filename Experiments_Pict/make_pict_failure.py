import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np


def read_data(dir_name, file_num):
    rewards = []
    for idx in range(1, file_num + 1):
        reward = []
        with open("{}/reward{}.txt".format(dir_name, idx), "r") as f:
            lines = f.readlines()
            for line in lines:
                num = re.findall("\-?\d+\.?\d*", line)
                reward.append(num)
            rewards.append(reward)

    total_reward = []
    for idx in range(rewards[0].__len__()):
        item = [0.0, 0.0]
        for reward in rewards:
            item[0] += float(reward[idx][0])
            item[1] += float(reward[idx][1])
        total_reward.append(item)
    for item in total_reward:
        item[0] /= rewards.__len__()
        item[1] /= rewards.__len__()
    ls1 = []
    ls2 = []
    for i in total_reward:
        ls1.append(i[0])
        ls2.append(i[1])
    return ls1, ls2


def read_data2(dir_name, file_num):
    lss = []
    for idx in range(1, file_num + 1):
        ls = []
        with open("{}/record new {}.txt".format(dir_name, idx), "r") as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    continue
                num = re.findall("\-?\d+\.?\d*", line)
                ls.append(int(float(num[4])))
            lss.append(ls)

    return lss


times = read_data2(r"time", 4)
# times[0] = times[0][10:14]
# times[1] = times[1][10:14]
# times[2] = times[2][10:14]
# times[3] = times[3][10:14]
# good1, bad1 = read_data(r"0.15bad\0.1", 3)
# good2, bad2 = read_data(r"0.15bad\0.2", 3)
# good3, bad3 = read_data(r"0.15bad\0.3", 3)
# good4, bad4 = read_data(r"0.15bad\0.4", 3)
# goods = [good1, good2, good3, good4]
# bads = [bad1, bad2, bad3, bad4]

# good1, bad1 = read_data(r"0.1 bad\0.2", 3)
# good1 = [5.0] + good1
# bad1 = [5.0] + bad1
# good2, bad2 = read_data(r"0.13bad\0.2", 3)
# good2 = [5.0] + good2
# bad2 = [5.0] + bad2
# good3, bad3 = read_data(r"0.15bad\0.2", 3)
# good3 = [5.0] + good3
# bad3 = [5.0] + bad3
# good4, bad4 = read_data(r"0.18bad\0.2", 3)
# good4 = [5.0] + good4
# bad4 = [5.0] + bad4
# good4, bad4 = read_data(r"0.1 bad\0.4", 3)

# good4 = [5.0] + good4
# bad4 = [5.0] + bad4
# good3 = [5.0] + good3
# bad3 = [5.0] + bad3
# good2 = [5.0] + good2
# bad2 = [5.0] + bad2
# good1 = [5.0] + good1
# bad1 = [5.0] + bad1
# goods = [good1, good2, good3, good4]
# bads = [bad1, bad2, bad3, bad4]

plt.figure('Draw')
# x_major_locator = MultipleLocator(10)
ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)

# plt.legend(loc='upper right')

# plt.plot(goods[0], color='red',
#          label="10% malicious nodes")  # plot绘制折线图
# plt.plot(bads[0], color='blue')
# plt.plot(goods[1], color='red', linestyle='--',
#          label="20% malicious nodes")  # plot绘制折线图
# plt.plot(bads[1], color='blue', linestyle='--')
# plt.plot(goods[2], color='red', linestyle=':',
#          label="30% malicious nodes")  # plot绘制折线图
# plt.plot(bads[2], color='blue', linestyle=':')
# plt.plot(goods[3], color='red', linestyle='-.',
#          label="40% malicious nodes")  # plot绘制折线图
# plt.plot(bads[3], color='blue', linestyle='-.')


# plt.plot(goods[0], color='red',
#          label=r'$\beta$ = 0.10')  # plot绘制折线图
# plt.plot(bads[0], color='blue')
# plt.plot(goods[1], color='red', linestyle='--',
#          label=r'$\beta$ = 0.13')  # plot绘制折线图
# plt.plot(bads[1], color='blue', linestyle='--')
# plt.plot(goods[2], color='red', linestyle=':',
#          label=r'$\beta$ = 0.15')  # plot绘制折线图
# plt.plot(bads[2], color='blue', linestyle=':')
# plt.plot(goods[3], color='red', linestyle='-.',
#          label=r'$\beta$ = 0.18')  # plot绘制折线图
# plt.plot(bads[3], color='blue', linestyle='-.')

x = list(range(0, len(times[0])))
xticklabels = [i * 5 for i in x]

# 设置y轴刻度和标签
y_ticks = [i * 1000 for i in range(3, 6)]        # 生成刻度值列表
# ax.set_yticks(y_ticks, minor=False)
# ax.set_ylim([0, 9000])          # 设置纵坐标范围

# 加个mark，颜色都不一样
# plt.plot(times[0], color='red',
#          label="3 points failure")  # plot绘制折线图
# plt.plot(times[1], color='red', linestyle='--',
#          label="2 points failure")  # plot绘制折线图
# plt.plot(times[2], color='red', linestyle=':',
#          label="1 point failure")  # plot绘制折线图
# plt.plot(times[3], color='red', linestyle='-.',
#          label="all nodes survive")  # plot绘制折线图
# x = list(range(50, len(times[0]) * 5 + 50, 5))  # 修改了x的范围和步长
x = list(range(0, len(times[0]) * 5 , 5))  # 修改了x的范围和步长


plt.plot(x, times[0], color='red',
         label="3 points failure", linewidth=3.0,marker=".", markersize=10)  # plot绘制折线图
plt.plot(x, times[1], color='darkviolet',
         label="2 points failure", linestyle=':', linewidth=3.0,marker=".", markersize=10)  # plot绘制折线图
plt.plot(x, times[2], color='lawngreen',
         label="1 point failure", linestyle='--', linewidth=3.0,marker=".", markersize=10)  # plot绘制折线图
plt.plot(x, times[3], color='blue',
         label="all nodes survive", linestyle='-.', linewidth=3.0,marker=".", markersize=10)  # plot绘制折线图

# plt.plot(x, times[0], color='red',
#          linewidth=3.0,marker=".", markersize=20)  # plot绘制折线图
# plt.plot(x, times[1], color='darkviolet',
#          linestyle=':', linewidth=3.0,marker=".", markersize=20)  # plot绘制折线图
# plt.plot(x, times[2], color='lawngreen',
#          linestyle='--', linewidth=3.0,marker=".", markersize=20)  # plot绘制折线图
# plt.plot(x, times[3], color='blue',
#          linestyle='-.', linewidth=3.0,marker=".", markersize=20)  # plot绘制折线图


# plt.xlabel('Communication Rounds', fontsize=25)
# plt.ylabel('Execution Time(s)', fontsize=25)
# plt.rcParams.update({'font.size': 25})
plt.tick_params(labelsize=25)
# ax.set_xticks(x, minor=False)
# ax.set_xticklabels(xticklabels, fontdict=None, minor=False)


plt.legend()
plt.draw()  # 显示绘图
plt.pause(5)  # 显示5秒
plt.savefig("pic1.jpg", dpi=600, bbox_inches='tight')  # 保存图象
plt.savefig("pic1.eps", dpi=600, bbox_inches='tight')  # 保存图象
plt.close()  # 关闭图表
