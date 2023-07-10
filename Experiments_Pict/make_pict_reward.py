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


# def read_data2(dir_name, file_num):
#     lss = []
#     for idx in range(1, file_num + 1):
#         ls = []
#         with open("{}/record new {}.txt".format(dir_name, idx), "r") as f:
#             lines = f.readlines()
#             for line in lines:
#                 if line[0] == '#':
#                     continue
#                 num = re.findall("\-?\d+\.?\d*", line)
#                 ls.append(int(float(num[4])))
#             lss.append(ls)

#     return lss


# times = read_data2(r"time", 4)


# good1, bad1 = read_data(r"0.15bad\0.1", 3)
# good2, bad2 = read_data(r"0.15bad\0.2", 3)
# good3, bad3 = read_data(r"0.15bad\0.3", 3)
# good4, bad4 = read_data(r"0.15bad\0.4", 3)
# goods = [good1, good2, good3, good4]
# bads = [bad1, bad2, bad3, bad4]

good1, bad1 = read_data(r"0.1 bad\0.1", 3)
# good1 = [5.0] + good1
# bad1 = [5.0] + bad1
good2, bad2 = read_data(r"0.13bad\0.1", 3)
# good2 = [5.0] + good2
# bad2 = [5.0] + bad2
good3, bad3 = read_data(r"0.15bad\0.1", 3)
# good3 = [5.0] + good3
# bad3 = [5.0] + bad3
good4, bad4 = read_data(r"0.18bad\0.1", 3)
# good4 = [5.0] + good4
# bad4 = [5.0] + bad4
# good4, bad4 = read_data(r"0.1 bad\0.4", 3)

good4 = [5.0] + good4
bad4 = [5.0] + bad4
good3 = [5.0] + good3
bad3 = [5.0] + bad3
good2 = [5.0] + good2
bad2 = [5.0] + bad2
good1 = [5.0] + good1
bad1 = [5.0] + bad1
goods = [good1, good2, good3, good4]
bads = [bad1, bad2, bad3, bad4]

plt.figure('Draw')
# x_major_locator = MultipleLocator(10)
ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)

plt.rcParams.update({'font.size': 15})
# plt.legend(loc='upper right')
x = list(range(0, len(goods[0])))
xticklabels = [i * 5 for i in x]
# plt.plot(x, goods[0], color='red',
#          label="10% malicious nodes", linestyle='-')  # plot绘制折线图
# plt.plot(x, goods[1], color='darkviolet',
#          label="20% malicious nodes", linestyle=':')  # plot绘制折线图
# plt.plot(x, goods[2], color='lawngreen',
#          label="30% malicious nodes", linestyle='--')  # plot绘制折线图
# plt.plot(x, goods[3], color='blue',
#          label="40% malicious nodes", linestyle='-.')  # plot绘制折线图

# plt.plot(x, bads[0], color='red',
#          label="10% malicious nodes", linestyle='-')  # plot绘制折线图
# plt.plot(x, bads[1], color='darkviolet',
#          label="20% malicious nodes", linestyle=':')  # plot绘制折线图
# plt.plot(x, bads[2], color='lawngreen',
#          label="30% malicious nodes", linestyle='--')  # plot绘制折线图
# plt.plot(x, bads[3], color='blue',
#          label="40% malicious nodes", linestyle='-.')  # plot绘制折线图


plt.plot(x, goods[0], color='red',
         label=r'$\beta_6$ = 0.10', linestyle='-')  # plot绘制折线图
plt.plot(x, goods[1], color='darkviolet',
         label=r'$\beta_6$ = 0.13', linestyle=':')  # plot绘制折线图
plt.plot(x, goods[2], color='lawngreen',
         label=r'$\beta_6$ = 0.15', linestyle='--')  # plot绘制折线图
plt.plot(x, goods[3], color='blue',
         label=r'$\beta_6$ = 0.18', linestyle='-.')  # plot绘制折线图

# plt.plot(x, bads[0], color='red',
#          label=r'$\beta_6$ = 0.10', linestyle='-')  # plot绘制折线图
# plt.plot(x, bads[1], color='darkviolet',
#          label=r'$\beta_6$ = 0.13', linestyle=':')  # plot绘制折线图
# plt.plot(x, bads[2], color='lawngreen',
#          label=r'$\beta_6$ = 0.15', linestyle='--')  # plot绘制折线图
# plt.plot(x, bads[3], color='blue',
#          label=r'$\beta_6$ = 0.18', linestyle='-.')  # plot绘制折线图

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


# 设置y轴刻度和标签
# y_ticks = [i * 1000 for i in range(1, 9)]        # 生成刻度值列表
# ax.set_yticks(y_ticks, minor=False)
# ax.set_ylim([0, 9000])          # 设置纵坐标范围

plt.rcParams.update({'font.size': 28})
plt.xlabel('communication rounds', fontsize=28)
plt.ylabel('reward', fontsize=28)
plt.tick_params(labelsize=28)
# ax.set_xticks(x, minor=False)
# ax.set_xticklabels(xticklabels, fontdict=None, minor=False)


plt.legend()
plt.draw()  # 显示绘图
plt.pause(5)  # 显示5秒
plt.savefig("pic1.jpg", dpi=600, bbox_inches='tight')  # 保存图象
plt.savefig("pic1.eps", dpi=600, bbox_inches='tight')  # 保存图象
plt.close()  # 关闭图表
# def reward_pict(rewards):
#     for i in total_reward:
#     plt.figure('Draw')
#     x_major_locator=MultipleLocator(1)
#     #ax为两条坐标轴的实例
#     ax=plt.gca()
#     #把x轴的主刻度设置为1的倍数
#     ax.xaxis.set_major_locator(x_major_locator)
#     plt.plot(p3,linestyle='dotted',label="the unoptimized gobang algorithm")  # plot绘制折线图
#     plt.plot(p4,label="the proposed algorithm")
#     plt.xlabel('steps')
#     plt.ylabel('calculating time (seconds)')
#     plt.legend()
#     plt.draw()  # 显示绘图
#     plt.pause(5)  #显示5秒
#     # plt.show()
#     plt.savefig("pic2.jpg")  #保存图象
#     plt.close()   #关闭图表
