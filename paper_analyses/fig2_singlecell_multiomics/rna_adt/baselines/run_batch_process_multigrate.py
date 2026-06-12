import os

# 定义数据集名称列表
datasets = ["2_GSE128639","7_zenodo6368128/LUNG", "7_zenodo6368128/PBMC",
            "12_GSE193181/P5", "12_GSE193181/P8"]

# 遍历每个数据集并执行脚本
for dataset in datasets:
    print(f"Processing {dataset}")
    os.system(f"python process_data_Multigrate.py {dataset}")