import torch

# 创建事件
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 在 CUDA 操作之前记录开始事件
start_event.record()

# CUDA 操作
a = torch.randn(10000, 10000, device='cuda')
b = torch.randn(10000, 10000, device='cuda')
c = a + b

# 在 CUDA 操作之后记录结束事件
end_event.record()

# 同步事件
end_event.synchronize()

# 计算并打印事件间的时间
elapsed_time = start_event.elapsed_time(end_event)
print(f"Elapsed time: {elapsed_time} ms")