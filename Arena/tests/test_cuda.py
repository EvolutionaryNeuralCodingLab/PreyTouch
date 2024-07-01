import config


def test_torch():
    try:
        import torch
    except Exception:
        print('PyTorch not found. Setting IS_GPU to False')
        config.IS_GPU = False


def test_cuda_gpu():
    import torch
    if not torch.cuda.is_available():
        print('CUDA GPU not found. Setting IS_GPU to False')
        config.IS_GPU = False