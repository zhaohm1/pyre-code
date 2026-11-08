"""Channel-wise Zipformer bypass exercise."""

TASK = {
    "title": "Zipformer Channel-wise Bypass",
    "title_zh": "Zipformer 逐通道 Bypass",
    "difficulty": "Easy",
    "description_en": "Blend x and y, both (B,T,D), with channel weights c (D,): return (1-c)*x+c*y using broadcasting. Assume 0<=c<=1. c=0 selects x, c=1 selects y. Keep c differentiable; do not clamp it. This is the core mixing equation only, without icefall's training schedule or parameter constraints.",
    "description_zh": "用逐通道权重 c (D,) 混合形状均为 (B,T,D) 的 x、y：返回 (1-c)*x+c*y，并正确广播。假定 0<=c<=1；c=0 选 x，c=1 选 y。保留 c 的梯度，不对它做截断。本题仅练习核心混合公式，不包含 icefall 的训练调度或参数约束。",
    "function_name": "zipformer_bypass",
    "hint": "Broadcast c over batch and time; blend per channel.",
    "hint_zh": "将 c 广播到批次与时间维，逐通道混合。",
    "tests": [
        {"name": "Per-channel choices and interpolation", "code": """\nimport torch\nx=torch.tensor([[[1.,2.,3.],[4.,5.,6.]]]); y=torch.tensor([[[9.,8.,7.],[6.,5.,4.]]]); c=torch.tensor([0.,0.25,1.])\nz={fn}(x,y,c)\nassert z.shape==x.shape and torch.allclose(z,torch.tensor([[[1.,3.5,7.],[4.,5.,4.]]]))\n"""},
        {"name": "Batch broadcasting and gradients", "code": """\nimport torch\nx=torch.randn(2,3,4,requires_grad=True); y=torch.randn(2,3,4,requires_grad=True); c=torch.tensor([0.2,0.4,0.6,0.8],requires_grad=True)\nz={fn}(x,y,c); z.square().sum().backward()\nassert z.shape==x.shape and all(t.grad is not None and torch.isfinite(t.grad).all() for t in (x,y,c))\nassert torch.allclose({fn}(x,y,torch.zeros(4)),x) and torch.allclose({fn}(x,y,torch.ones(4)),y)\n"""},
    ],
    "solution": '''def zipformer_bypass(x, y, c):
    return (1 - c) * x + c * y''',
    "demo": """x=torch.zeros(1,2,3); y=torch.ones_like(x)\nprint(zipformer_bypass(x,y,torch.tensor([0.,0.5,1.])))""",
}
