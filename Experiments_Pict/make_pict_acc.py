import re
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np


file_num = 2
rewards = []
for idx in range(1, file_num + 1):
    reward = []
    with open("acc/record run {}.txt".format(idx), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '#':
                continue
            num = re.findall("\d+\.?\d*", line)
            reward.append(float(num[2])*100)
        rewards.append(reward)
rewards[0] = rewards[0][1:]
rewards[1] = rewards[1][1:]
# ls = []
# idx = 0
# for i in rewards[0]:
#     # i = float(i)*100.0
#     # reward[0][idx] = i
#     ls.append(i+float(i)*int(random.randint(-10, 10))*0.01*(18-idx)*0.1)
#     idx += 1
# print(type(i))

# idx = 0
# for i in range(len(rewards[0])):
#     print(idx)
#     print("{:.2f}".format(rewards[0][i]))
#     print("{:.2f}".format(rewards[1][i]))
#     idx += 5
# print(rewards[0])
# exit()
plt.figure('Draw')
# x_major_locator = MultipleLocator(10)
ax = plt.gca()
# ax.set_xlim(0, 20)
ax.set_ylim(55, 85)
# ax.set_ylim(25, 70)
# ax.set_ylim(75.5, 80)
# ax.xaxis.set_major_locator(x_major_locator)
# plt.plot(rewards[0], color='red',
#          label="honest nodes in 20% malicious nodes and 50\% attack probability")  # plot绘制折线图
# plt.plot(ls, color='blue', label="Average reward for malicious nodes")
x = list(range(5, len(rewards[0]) * 5 + 5, 5))  # 修改了x的范围和步长
plt.plot(x, rewards[0], color='red',
         label="FL", marker=".", markersize=10, linestyle='-',)  # plot绘制折线图
plt.plot(x, rewards[1], color='blue',
         label="BC-DFL", marker=".", markersize=10, linestyle='--',)

# plt.plot(x, rewards[0], color='red',
#          marker=".")  # plot绘制折线图
# plt.plot(x, rewards[1], color='blue',
#          marker=".")

plt.rcParams.update({'font.size': 20})
plt.tick_params(labelsize=20)
plt.xlabel('Communication Round', fontsize=20)
plt.ylabel('Test Accuracy(%)', fontsize=20)
# plt.legend(bbox_to_anchor=(0.5, -0.19), loc=8, ncol=9, frameon=False)
# plt.legend(loc=2)
plt.legend()
plt.draw()  # 显示绘图
plt.pause(5)  # 显示5秒
plt.savefig("pic1.jpg", dpi=600, bbox_inches='tight')  # 保存图象
plt.savefig("pic1.eps", dpi=600, bbox_inches='tight')  # 保存图象
plt.close()  # 关闭图表
