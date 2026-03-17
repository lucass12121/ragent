import sys

def check():
    errors = []

    # Python version
    v = sys.version_info
    print(f"Python: {v.major}.{v.minor}.{v.micro}")
    if v.major != 3 or v.minor < 10:
        errors.append("需要 Python >= 3.10")

    # torch
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"PyTorch: {torch.__version__}  CUDA: {cuda}")
        if cuda:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        errors.append("torch 未安装")

    # transformers
    try:
        import transformers
        print(f"transformers: {transformers.__version__}")
    except ImportError:
        errors.append("transformers 未安装")

    # datasets
    try:
        import datasets
        print(f"datasets: {datasets.__version__}")
    except ImportError:
        errors.append("datasets 未安装")

    # gradio
    try:
        import gradio
        print(f"gradio: {gradio.__version__}")
    except ImportError:
        errors.append("gradio 未安装")

    # verl (optional, only on GPU server)
    has_verl = False
    try:
        import verl
        has_verl = True
        print(f"verl: {verl.__version__}")
    except ImportError:
        pass

    # vllm (optional, only on GPU server)
    try:
        import vllm
        print(f"vllm: {vllm.__version__}")
    except ImportError:
        pass

    if errors:
        print("\n[FAIL] 环境检查未通过:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Determine environment type
    try:
        cuda_ok = torch.cuda.is_available()
    except NameError:
        cuda_ok = False

    if cuda_ok and has_verl:
        print("\n[OK] GPU环境就绪")
    else:
        print("\n[OK] 本地环境就绪")


if __name__ == "__main__":
    check()
