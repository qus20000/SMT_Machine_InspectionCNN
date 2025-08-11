import torch

print("CUDA 사용 가능:", torch.cuda.is_available())
print("현재 사용 중인 디바이스:", torch.cuda.current_device())
print("디바이스 이름:", torch.cuda.get_device_name(0))