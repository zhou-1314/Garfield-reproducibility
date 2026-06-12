import os

# 定义数据集名称列表
human_datasets = ["5_human_brain_10x", "6_pbmc_granulocyte_sorted_10x", "7_human_pbmc_10x",
                  "8_pbmc_unsorted_10k", "9_pbmc_unsorted_3k"]

mouse_datasets = ["1_ShareSeq_Skin", "2_brain_ShareSeq", "3_brain_SNARE","4_brain_ISSAAC_seq","10_GSE201402_down"]

# 遍历每个数据集并执行脚本
# for dataset in datasets:
#     param1 = "value1"
#     param2 = "value2"
#     param3 = "value3"
#
#     print(f"Processing {dataset} with parameters {param1}, {param2}, {param3}")
#     os.system(f"python process_data_Garfield.py {dataset} {param1} {param2} {param3}")

import subprocess
for dataset in human_datasets:
    param1 = dataset
    param2 = "hg38"

    print(f"Processing {dataset} with parameters {param1}, {param2}")
    subprocess.run(["python", "process_data_Garfield.py", param1, param2])

for dataset in mouse_datasets:
    param1 = dataset
    param2 = "mm10"

    print(f"Processing {dataset} with parameters {param1}, {param2}")
    subprocess.run(["python", "process_data_Garfield.py", param1, param2])
