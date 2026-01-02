import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from models.yolo import Model
import torch

def verify():
    print("Verifying custom module integration...")
    try:
        # 尝试初始化模型
        print("Loading model from cfg/training/train.yaml...")
        model = Model(cfg='cfg/training/train.yaml', ch=3, nc=80)
        print("Model initialization successful!")
        
        # 创建虚拟输入进行前向传播测试
        print("Running forward pass with dummy input...")
        dummy_input = torch.zeros(1, 3, 640, 640)
        _ = model(dummy_input)
        print("Forward pass successful!")
        
        print("VERIFICATION PASSED")
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
