import torch
import subprocess

def check_cuda_configuration():
    # 检查 PyTorch 是否检测到 CUDA
    if torch.cuda.is_available():
        print("CUDA 可用!")
        device_count = torch.cuda.device_count()
        print(f"检测到 {device_count} 个 GPU 设备。")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"设备 {i}: {device_name}")
    else:
        print("CUDA 不可用！")
        print("请检查以下配置：")
        print("1. 是否安装了 NVIDIA GPU 以及对应的 NVIDIA 驱动？")
        print("2. CUDA Toolkit 是否正确安装？")
        print("3. 是否安装了支持 GPU 版本的 PyTorch？")
        
        # 尝试运行 nvidia-smi 查看 GPU 驱动信息
        try:
            print("\n尝试调用 nvidia-smi 命令获取详细信息：")
            output = subprocess.check_output("nvidia-smi", shell=True)
            print(output.decode("utf-8"))
        except Exception as e:
            print("调用 nvidia-smi 命令失败。请确保 NVIDIA 驱动已安装并配置正确。")
            print("错误信息：", e)

if __name__ == "__main__":
    check_cuda_configuration()

# py3.7_cuda101_cudnn7_0