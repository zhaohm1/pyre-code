"""Pedagogical multi-rate temporal resampling exercise."""

TASK = {
    "title": "Zipformer Temporal Downsample and Restore",
    "title_zh": "Zipformer 时间维降采样与恢复",
    "difficulty": "Easy",
    "description_en": "Given x (B,T,D) and positive integer factor r, split time into consecutive groups of r, zero-pad the final group, but average only its valid frames. Return `(low, restored)`, where low is (B,ceil(T/r),D), and restored is (B,T,D) from repeating each low frame r times then trimming to T. T>=1. This is a teaching-only multi-rate operation, not icefall's learned down/up-sampling, skip connections or full U-Net encoder.",
    "description_zh": "给定 x (B,T,D) 和正整数 r，将时间维按 r 个连续帧分组；末组不足 r 时补零，但平均时只计有效帧。返回 `(low, restored)`：low 为 (B,ceil(T/r),D)，restored 为将每个低帧重复 r 次再裁剪到 T 的 (B,T,D)。T>=1。本题只是教学用多尺度运算，不等同于 icefall 的可学习降/升采样、跳跃连接或完整 U-Net 编码器。",
    "function_name": "zipformer_resample",
    "hint": "Use ceil division for the number of groups; the last group's denominator is its actual number of valid frames.",
    "hint_zh": "组数用向上取整；末组除以真实有效帧数。",
    "tests": [
        {"name": "Non-divisible sequence and restoration", "code": """\nimport torch\nx=torch.tensor([[[1.,10.],[3.,20.],[5.,30.],[7.,40.],[9.,50.]]])\nlow,up={fn}(x,2)\nassert torch.allclose(low,torch.tensor([[[2.,15.],[6.,35.],[9.,50.]]]))\nassert torch.allclose(up,torch.tensor([[[2.,15.],[2.,15.],[6.,35.],[6.,35.],[9.,50.]]]))\n"""},
        {"name": "Factor one and factor larger than time", "code": """\nimport torch\nx=torch.arange(12.).reshape(2,3,2)\na,b={fn}(x,1); assert torch.allclose(a,x) and torch.allclose(b,x)\na,b={fn}(x,5); assert a.shape==(2,1,2) and b.shape==x.shape\nassert torch.allclose(a,x.mean(1,keepdim=True)) and torch.allclose(b,a.expand_as(x))\n"""},
        {"name": "Gradient through both outputs", "code": """\nimport torch\nx=torch.randn(2,5,3,requires_grad=True); a,b={fn}(x,2); (a.square().sum()+b.square().sum()).backward()\nassert x.grad is not None and torch.isfinite(x.grad).all() and (x.grad.abs().sum(-1)>0).all()\n"""},
    ],
    "solution": '''def zipformer_resample(x, factor):
    import torch.nn.functional as F
    t = x.shape[1]
    groups = (t + factor - 1) // factor
    padded = F.pad(x, (0, 0, 0, groups * factor - t))
    sums = padded.reshape(x.shape[0], groups, factor, x.shape[2]).sum(2)
    counts = torch.arange(groups, device=x.device) * factor
    counts = (t - counts).clamp(min=1, max=factor).to(x.dtype)
    low = sums / counts[None, :, None]
    return low, low.repeat_interleave(factor, dim=1)[:, :t]''',
    "demo": """x=torch.arange(15.).reshape(1,5,3)\nprint([z.shape for z in zipformer_resample(x,2)])""",
}
