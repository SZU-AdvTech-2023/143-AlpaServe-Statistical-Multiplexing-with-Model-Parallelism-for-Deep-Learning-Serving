import argparse
import warnings

import os
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from alpa_serve.util import write_tsv
from benchmarks.alpa.equal_model_case import read_equal_model_case_tsv
from benchmarks.alpa.general_model_case import read_general_model_case_tsv
from benchmarks.alpa.plot_various_metrics import show_name, method2color, method2order

linestyles = ["solid", "dashed", "dashdot", "dotted"]
methodcolors = ["C2", "C1", "C0", "red"]

###modified
_DIFF_HEADS = ("title","xlabel","x-value","y-value-before","y-value-after","y-diff")
values = []
###
def plot_diff_goodput(title,xlabel,xs,ys_before,ys_after,output_file):
    output_file = output_file.replace(".tsv", title+"_diff.pdf")
    # 创建图表
    plt.figure()

    # 计算 y_after 与 y_before 的差值
    ys_diff = [y_after - y_before for y_after, y_before in zip(ys_after, ys_before)]

    # 绘制差值线条
    plt.plot(xs, ys_diff, label='Difference')

    # 添加标题和轴标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Difference in '+xlabel)

    # 添加图例
    plt.legend()

    # 保存为PDF
    plt.savefig(output_file, format='pdf')

    print(f"diff chart saved as {output_file}")


def plot_goodput_common_cmp(azure_data_before, azurev2_data_after, threshold, increasing, xlabel, output, ybottem, plot_legend=False):
    fig, axs = plt.subplots(1, len(azure_data_before) )
    titles = ["S1 @ MAF1", "S2 @ MAF1", "S3 @ MAF1", "S1 @ MAF2", "S2 @ MAF2", "S3 @ MAF2"]
    output_file = output.replace(".pdf", ".tsv")

    for data_before,data_after, ax, title in zip(azure_data_before , azurev2_data_after, axs, titles):
        methods = list(data_before.keys())
        if "mp-search" in methods and "mp-search-sep" in methods:
            methods.remove("mp-search")
        methods.sort(key=lambda x: method2order(x))

        curves = []
        legends = []
        # first_good = []
        x_max = 0
        y_max = 0
        for i, method in enumerate(methods):

            curve_before = data_before[method]

            xs_, ys_ = zip(*curve_before.items())
            xs = [x for x, _ in sorted(zip(xs_, ys_))]
            ys_before = [y for _, y in sorted(zip(xs_, ys_))]
            ys_before = np.array(ys_before) * 100
            curve_before = ax.plot(xs, ys_before, color=methodcolors[0], marker='.', linestyle=linestyles[0], linewidth=4,
                            markersize=15)
            curves.append(curve_before[0])
            legends.append(show_name("before"))
            # if increasing:
            #     iterator = range(len(xs))
            # else:
            #     iterator = reversed(range(len(xs)))

            # found = False
            # for i in iterator:
            #     if ys[i] >= threshold * 100:
            #         first_good.append(xs[i])
            #         found = True
            #         break
            # if not found:
            #     first_good.append(0)
            x_max = max(x_max, *xs)
            y_max = max(y_max, *ys_before)

            ########################################
            curve_after = data_after[method]

            xs_, ys_ = zip(*curve_after.items())
            xs = [x for x, _ in sorted(zip(xs_, ys_))]
            ys_after = [y for _, y in sorted(zip(xs_, ys_))]
            ys_after = np.array(ys_after) * 100
            curve_after = ax.plot(xs, ys_after, color=methodcolors[1], marker='.', linestyle=linestyles[1], linewidth=4,
                            markersize=15)
            curves.append(curve_after[0])
            legends.append(show_name("after"))


            for x,y_before,y_after in (zip(xs,ys_before,ys_after)):
                values = (title,xlabel,x,y_before,y_after,y_after-y_before)
                if output_file is not None:
                    write_tsv(_DIFF_HEADS, values, output_file)

            plot_diff_goodput(title,xlabel,xs,ys_before,ys_after,output_file)

            # if increasing:
            #     iterator = range(len(xs))
            # else:
            #     iterator = reversed(range(len(xs)))

            # found = False
            # for i in iterator:
            #     if ys[i] >= threshold * 100:
            #         first_good.append(xs[i])
            #         found = True
            #         break
            # if not found:
            #     first_good.append(0)

            x_max = max(x_max, *xs)
            y_max = max(y_max, *ys_after)

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        ax.set_ylim(bottom=ybottem, top=max(y_max * 1.02, 100))
        ax.grid()

        ax.set_xlabel(xlabel, fontsize=20)
        # ax.set_ylabel("Workload Satisfaction (%)")
        if plot_legend:
            ax.set_title(title, fontsize=20)

        # for i in range(len(methods)):
        #     if first_good[i] == 0:
        #         continue
        #     ax.axvline(first_good[i], color=methodcolors[i], linestyle=":", linewidth=4)

    fig.text(0.1, 0.5, "SLO Attainment (%)", va='center', rotation='vertical', fontsize=20)

    if plot_legend:
        fig.legend(curves, legends, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.2), fontsize=20)

    figure_size = (40, 4)
    fig.set_size_inches(figure_size)
    fig.savefig(output, bbox_inches='tight')
    print(f"Output the plot to {output}")

def plot_goodput_vs_num_devices_cmp(azure_lines_before, azure_lines_after, threshold, folder, pdf, ybottom, plot_legend=False):
    # Dict[policy -> Dict[num_devices -> goodput]]
    azure_data_before = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azure_lines_before))]
    azure_data_after = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azure_lines_after))]

    for data, lines in zip(azure_data_before, azure_lines_before):
        for line in lines:
            if line["exp_name"] != "goodput_vs_num_devices" or "mp-search" not in line["policy_name"]:
                continue

            policy, x, goodput = (
                line["policy_name"], line["num_devices"], line["goodput"])
            data[policy][x] = goodput

    for data, lines in zip(azure_data_after, azure_lines_after):
        for line in lines:
            if line["exp_name"] != "goodput_vs_num_devices" or "mp-search" not in line["policy_name"]:
                continue

            policy, x, goodput = (
                line["policy_name"], line["num_devices"], line["goodput"])
            data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_num_devices.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_num_devices.png")

    plot_goodput_common_cmp(azure_data_before, azure_data_after, threshold, True, "#devices", output, ybottom, plot_legend)

def plot_goodput_vs_rate_scale_cmp(azure_lines_before, azure_lines_after, threshold, folder, pdf, ybottom, plot_legend=False):
    # Dict[policy -> Dict[num_devices -> goodput]]
    azure_data_before = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azure_lines_before))]
    azure_data_after = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azure_lines_after))]

    for data, lines in zip(azure_data_before, azure_lines_before):
        for line in lines:
            if line["exp_name"] != "goodput_vs_rate_scale" or "mp-search" not in line["policy_name"]:
                continue

            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["rate_scale"], line["goodput"])
            data[policy][x] = goodput

    for data, lines in zip(azure_data_after, azure_lines_after):
        for line in lines:
            if line["exp_name"] != "goodput_vs_rate_scale" or "mp-search" not in line["policy_name"]:
                continue

            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["rate_scale"], line["goodput"])
            data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_rate_scale.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_rate_scale.png")

    plot_goodput_common_cmp(azure_data_before, azure_data_after, threshold, False, "Rate Scale", output, ybottom, plot_legend)

def plot_goodput_vs_cv_scale_cmp(azure_lines_before, azure_lines_after, threshold, folder, pdf, ybottom, plot_legend=False):
    # Dict[policy -> Dict[num_devices -> goodput]]
    azure_data_before = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azure_lines_before))]
    azure_data_after = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azure_lines_after))]

    for data, lines in zip(azure_data_before, azure_lines_before):
        for line in lines:
            if line["exp_name"] != "goodput_vs_cv_scale" or "mp-search" not in line["policy_name"]:
                continue

            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["cv_scale"], line["goodput"])
            data[policy][x] = goodput

    for data, lines in zip(azure_data_after, azure_lines_after):
        for line in lines:
            if line["exp_name"] != "goodput_vs_cv_scale" or "mp-search" not in line["policy_name"]:
                continue

            policy, x, goodput = (
                line["policy_name"], line["arrival_process_kwargs"]["cv_scale"], line["goodput"])
            data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_cv_scale.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_cv_scale.png")

    plot_goodput_common_cmp(azure_data_before, azure_data_after, threshold, False, "CV Scale", output, ybottom, plot_legend)

def plot_goodput_vs_slo_cmp(azure_lines_before, azure_lines_after, threshold, folder, pdf, ybottom, plot_legend=False):
    # Dict[policy -> Dict[num_devices -> goodput]]
    azure_data_before = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azure_lines_before))]
    azure_data_after = [defaultdict(lambda: defaultdict(dict)) for _ in range(len(azure_lines_after))]

    for data, lines in zip(azure_data_before, azure_lines_before):
        for line in lines:
            if line["exp_name"] != "goodput_vs_slo" or "mp-search" not in line["policy_name"]:
                continue

            policy, x, goodput = (
                line["policy_name"], line["slo_scale"], line["goodput"])
            data[policy][x] = goodput

    for data, lines in zip(azure_data_after, azure_lines_after):
        for line in lines:
            if line["exp_name"] != "goodput_vs_slo" or "mp-search" not in line["policy_name"]:
                continue

            policy, x, goodput = (
                line["policy_name"], line["slo_scale"], line["goodput"])
            data[policy][x] = goodput

    if pdf:
        output = os.path.join(folder, "goodput_vs_slo.pdf")
    else:
        output = os.path.join(folder, "goodput_vs_slo.png")

    plot_goodput_common_cmp(azure_data_before, azure_data_after, threshold, True, "SLO Scale", output, ybottom, plot_legend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="sec6_2_data_cmp")
    parser.add_argument("--output-dir", type=str, default="paper_figures_cmp")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    threshold = 0.99

    bert1dot3_azurev1_before = args.input + "/azure_v1_1dot3b_before.tsv"
    bert6dot7_azurev1_before = args.input + "/azure_v1_6dot7b_before.tsv"
    mixed_azurev1_before = args.input + "/azure_v1_mixed_before.tsv"
    bert1dot3_azurev2_before = args.input + "/azure_v2_1dot3b_before.tsv"
    bert6dot7_azurev2_before = args.input + "/azure_v2_6dot7b_before.tsv"
    mixed_azurev2_before = args.input + "/azure_v2_mixed_before.tsv"

    bert1dot3_azurev1_after = args.input + "/azure_v1_1dot3b_after.tsv"
    bert6dot7_azurev1_after = args.input + "/azure_v1_6dot7b_after.tsv"
    mixed_azurev1_after = args.input + "/azure_v1_mixed_after.tsv"
    bert1dot3_azurev2_after = args.input + "/azure_v2_1dot3b_after.tsv"
    bert6dot7_azurev2_after = args.input + "/azure_v2_6dot7b_after.tsv"
    mixed_azurev2_after = args.input + "/azure_v2_mixed_after.tsv"

    ## 获取的是每一列表格的数据
    azure_lines_before = [read_equal_model_case_tsv(bert1dot3_azurev1_before),
                          read_equal_model_case_tsv(bert6dot7_azurev1_before),
                          read_general_model_case_tsv(mixed_azurev1_before),
                          read_equal_model_case_tsv(bert1dot3_azurev2_before),
                          read_equal_model_case_tsv(bert6dot7_azurev2_before),
                          read_general_model_case_tsv(mixed_azurev2_before)]

    azure_lines_after = [read_equal_model_case_tsv(bert1dot3_azurev1_after),
                         read_equal_model_case_tsv(bert6dot7_azurev1_after),
                         read_general_model_case_tsv(mixed_azurev1_after),
                         read_equal_model_case_tsv(bert1dot3_azurev2_after),
                         read_equal_model_case_tsv(bert6dot7_azurev2_after),
                         read_general_model_case_tsv(mixed_azurev2_after)]

    plot_goodput_vs_num_devices_cmp(azure_lines_before,azure_lines_after,threshold,args.output_dir,args.pdf,10,True)

    plot_goodput_vs_rate_scale_cmp(azure_lines_before, azure_lines_after, threshold, args.output_dir, args.pdf, 10)
    plot_goodput_vs_cv_scale_cmp(azure_lines_before, azure_lines_after, threshold, args.output_dir, args.pdf, 10)
    plot_goodput_vs_slo_cmp(azure_lines_before, azure_lines_after, threshold, args.output_dir, args.pdf, 0)
