import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
torch.cuda.set_device(0)
print(torch.cuda.get_device_name(1))
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())
print(torch.cuda.max_memory_allocated())
print(torch.cuda.current_device())