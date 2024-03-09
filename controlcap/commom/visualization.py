import matplotlib.pyplot as plt
import numpy as np
import math

figs = ["fig1", "fig2"]

if "fig1" in figs:
    # 生成一些示例数据
    x = [0, 0.1, 0.5, 0.8, 1]
    y1 = [15, 42.09, 42.28, 42.42, 42.45]
    # y2 = [66.16, 66.25, 65.87, 65.79, 42.45]

    # 创建一个图形对象和一个坐标轴对象
    fig, ax = plt.subplots(figsize=(7, 4))

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')  # 设置边框颜色
        spine.set_linewidth(1.5)

    # 绘制普通折线图
    size =20
    plt.rc("font", size=size)
    plt.xticks(size=size)
    plt.yticks(size=size)
    ax.plot(x, y1, label='without guidance', marker="o", linewidth=3)

    for a, b in zip(x, y1):
        plt.text(a, b, b, fontsize=size-5, va='bottom', ha='center')

    # ax.plot(x, y2, label='with guidance')

    base1 = 42
    base2 = 43
    line1 = 0.1
    line2 = 10
    _base1 = base1 * line1
    _base2 = (base2 - base1) * line2 + _base1

    def forward(x):
        outx = np.zeros_like(x)
        # x = x - base
        # mask = x>=0
        # outx[mask] = (x[mask] + bias) ** (1/ratio) + base - bias ** (1/ratio)
        # outx[~mask] = -(-x[~mask] + bias) ** (1/ratio) + base + bias ** (1/ratio)
        # print("111")
        mask1 = x < base1
        mask2 = (x >= base1) & (x < base2)
        mask3 = x >= base2

        outx[mask1] = line1 * x[mask1]
        outx[mask2] = line2 * (x[mask2] - base1) + _base1
        outx[mask3] = line1 * (x[mask3] - base2) + _base2



        return outx


    def inverse(x):
        outx = np.zeros_like(x)
        # x = x - base
        # mask = x >= 0
        # outx[mask] = (x[mask] + bias ** (1/ratio)) ** ratio + base - bias
        # outx[~mask] = -(x[~mask] - bias ** (1/ratio)) ** ratio + base + bias
        mask1 = x < _base1
        mask2 = (x >= _base1) & (x < _base2)
        mask3 = x >= _base2

        outx[mask1] = x[mask1] / line1
        outx[mask2] = (x[mask2] - _base1) / line2 + base1
        outx[mask3] = (x[mask3] - _base2) / line2 + base2



        return outx


    # 使用对数轴
    # custom_y_ticks = [10, 40, 41, 42, 43, 65, 66, 67]
    # ax.set_yticks(custom_y_ticks)

    ax.set_yscale('function', functions=(forward, inverse))

    # 添加网格和标签
    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax.legend()
    # ax.set_xlabel('drop probability')
    # ax.set_ylabel('mAP(%)')
    # ax.set_title('Line Plot with Logarithmic Scale')

    # 显示图形
    # plt.show()
    plt.savefig("drop_ratio_fig1.png")


if "fig2" in figs:
    # 生成一些示例数据
    x = [0, 0.1, 0.5, 0.8, 1]
    # y1 = [15, 42.09, 42.28, 42.42, 42.45]
    y2 = [66.16, 66.25, 65.87, 65.79, 42.45]

    # 创建一个图形对象和一个坐标轴对象
    fig, ax = plt.subplots(figsize=(7, 4))

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('black')  # 设置边框颜色
        spine.set_linewidth(1.5)

    # 绘制普通折线图
    size = 20
    plt.rc("font", size=size)
    plt.xticks(size=size)
    plt.yticks(size=size)
    plt.legend(loc='lower left')
    ax.plot(x, y2, label='with guidance', marker="o", linewidth=3, color="green")
    # ax.plot(x, y2, label='with guidance')

    for a, b in zip(x, y2):
        plt.text(a, b, b, fontsize=size-5, va='bottom', ha='center')

    base1 = 65.5
    base2 = 66.5
    line1 = 0.1
    line2 = 5
    _base1 = base1 * line1
    _base2 = (base2 - base1) * line2 + _base1


    def forward(x):
        outx = np.zeros_like(x)
        # x = x - base
        # mask = x>=0
        # outx[mask] = (x[mask] + bias) ** (1/ratio) + base - bias ** (1/ratio)
        # outx[~mask] = -(-x[~mask] + bias) ** (1/ratio) + base + bias ** (1/ratio)
        # print("111")
        mask1 = x < base1
        mask2 = (x >= base1) & (x < base2)
        mask3 = x >= base2

        outx[mask1] = line1 * x[mask1]
        outx[mask2] = line2 * (x[mask2] - base1) + _base1
        outx[mask3] = line1 * (x[mask3] - base2) + _base2

        return outx


    def inverse(x):
        outx = np.zeros_like(x)
        # x = x - base
        # mask = x >= 0
        # outx[mask] = (x[mask] + bias ** (1/ratio)) ** ratio + base - bias
        # outx[~mask] = -(x[~mask] - bias ** (1/ratio)) ** ratio + base + bias
        mask1 = x < _base1
        mask2 = (x >= _base1) & (x < _base2)
        mask3 = x >= _base2

        outx[mask1] = x[mask1] / line1
        outx[mask2] = (x[mask2] - _base1) / line2 + base1
        outx[mask3] = (x[mask3] - _base2) / line2 + base2

        return outx


    # 使用对数轴
    # custom_y_ticks = [10, 40, 41, 42, 43, 65, 66, 67]
    # ax.set_yticks(custom_y_ticks)

    ax.set_yscale('function', functions=(forward, inverse))

    # 添加网格和标签
    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax.legend()
    # ax.set_xlabel('drop probability')
    # ax.set_ylabel('mAP(%)')
    # ax.set_title('Line Plot with Logarithmic Scale')

    # 显示图形
    # plt.show()
    plt.savefig("drop_ratio_fig2.png")
