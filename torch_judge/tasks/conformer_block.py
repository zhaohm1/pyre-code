"""Conformer block ordering and half-step residual exercise."""

TASK = {
    "title": "Conformer Macaron Block",
    "title_zh": "Conformer Macaron 块",
    "difficulty": "Medium",
    "description_en": "Compose callables `ffn1`, `attention`, `conv`, `ffn2` on x (B,T,D): a=x+0.5*ffn1(x); b=a+attention(a); c=b+conv(b); return layer_norm(c+0.5*ffn2(c), normalized_shape=D) without affine parameters. Each callable returns the same shape. This tests order and half-step residuals; the callables stand in for full modules and no masks/dropout are included.",
    "description_zh": "对 x (B,T,D) 依次组合可调用对象 `ffn1`、`attention`、`conv`、`ffn2`：a=x+0.5*ffn1(x)，b=a+attention(a)，c=b+conv(b)，返回无仿射参数的 layer_norm(c+0.5*ffn2(c), normalized_shape=D)。各模块保持形状。本题检验顺序和半步残差；可调用对象代替完整模块，不包含掩码或 dropout。",
    "function_name": "conformer_block",
    "hint": "Keep the four intermediate states distinct; normalize only after the second half-step residual.",
    "hint_zh": "保留四个中间状态；只在第二个半步残差之后归一化。",
    "tests": [
        {"name": "Order and half-step arithmetic", "code": """\nimport torch, torch.nn.functional as F\nx=torch.tensor([[[1.,2.,4.],[2.,3.,7.]]])\nf1=lambda z:z*2; att=lambda z:z+1; conv=lambda z:z.flip(-1); f2=lambda z:z*3\ny={fn}(x,f1,att,conv,f2)\na=x+0.5*f1(x); b=a+att(a); c=b+conv(b); expected=F.layer_norm(c+0.5*f2(c),(3,))\nassert torch.allclose(y,expected,atol=1e-6)\n"""},
        {"name": "Final normalization and gradients", "code": """\nimport torch\nx=torch.randn(2,4,5,requires_grad=True)\ny={fn}(x,lambda z:z,lambda z:z*0.1,lambda z:z.square(),lambda z:z*0.3)\nassert y.shape==x.shape and torch.allclose(y.mean(-1),torch.zeros(2,4),atol=1e-5)\ny.square().sum().backward(); assert x.grad is not None and torch.isfinite(x.grad).all()\n"""},
    ],
    "solution": '''def conformer_block(x, ffn1, attention, conv, ffn2):
    import torch.nn.functional as F
    a = x + 0.5 * ffn1(x)
    b = a + attention(a)
    c = b + conv(b)
    return F.layer_norm(c + 0.5 * ffn2(c), (x.shape[-1],))''',
    "demo": """x=torch.randn(1,3,4); zero=lambda z:torch.zeros_like(z)\nprint(conformer_block(x,zero,zero,zero,zero).shape)""",
}
