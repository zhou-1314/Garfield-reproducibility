import os

# 定义数据集名称列表
datasets = [#"1_ShareSeq_Skin", "2_brain_ShareSeq", "3_brain_SNARE",
            "4_brain_ISSAAC_seq",
            "5_human_brain_10x", "6_pbmc_granulocyte_sorted_10x", "7_human_pbmc_10x",
            "8_pbmc_unsorted_10k", "9_pbmc_unsorted_3k", "10_GSE201402_down"]

# 遍历每个数据集并执行脚本
for dataset in datasets:
    print(f"Processing {dataset}")
    os.system(f"python process_data_Multigrate.py {dataset}")