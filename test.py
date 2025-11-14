import torch
import transformers

print(f"PyTorch版本: {torch.__version__}")
print(f"Transformers版本: {transformers.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 如果PyTorch版本低于2.0，建议升级
if int(torch.__version__.split('.')[0]) < 2:
    print("建议升级PyTorch到2.0+版本")