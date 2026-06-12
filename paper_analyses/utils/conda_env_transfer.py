import os
import tarfile
import sys


def unpack_env_tar(env_name: str, tar_file_path: str, conda_envs_path: str = '/home/zhouwg/software/anaconda3/envs'):
    """
    解压指定路径下的 tar.gz 文件并创建新的 Conda 环境目录。

    :param env_name: 要解压的环境名称
    :param tar_file_path: tar.gz 文件的完整路径
    :param conda_envs_path: Conda 环境存放的根路径，默认为 '/home/username/anaconda3/envs'
    """
    # 创建新环境路径
    new_env_path = os.path.join(conda_envs_path, env_name)
    if not os.path.exists(new_env_path):
        os.mkdir(new_env_path)

    # 解压 tar.gz 文件
    with tarfile.open(tar_file_path) as t_file:
        t_file.extractall(new_env_path)

    print(f"Environment '{env_name}' has been unpacked to '{new_env_path}'")


# 检查是否提供了正确的命令行参数
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <env_name> <tar_file_path> [conda_envs_path]")
        sys.exit(1)

    # 获取命令行参数
    env_name = sys.argv[1]
    tar_file_path = sys.argv[2]
    conda_envs_path = sys.argv[3] if len(sys.argv) > 3 else '/home/zhouwg/software/anaconda3/envs'

    # 调用函数解压环境
    unpack_env_tar(env_name, tar_file_path, conda_envs_path)