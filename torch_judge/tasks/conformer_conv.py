"""Simplified Conformer temporal convolution exercise."""

TASK = {
    "title": "Conformer GLU Depthwise Convolution",
    "title_zh": "Conformer GLU 与逐通道卷积",
    "difficulty": "Medium",
    "description_en": "Given `x` (B,T,D), `pointwise` (D,2D), `depthwise` (D,K), and `out_weight` (D,D), compute z=x@pointwise, GLU=z[:,:,:D]*sigmoid(z[:,:,D:]), same-length temporal depthwise cross-correlation with zero padding K//2 on each side, then output=conv@out_weight. K is positive and odd; T>=1. Each channel uses only its own kernel. This focused exercise omits Conformer's normalization, Swish, dropout and residual.",
    "description_zh": "给定 `x` (B,T,D)、`pointwise` (D,2D)、`depthwise` (D,K)、`out_weight` (D,D)，依次计算 z=x@pointwise、GLU=z[:,:,:D]*sigmoid(z[:,:,D:])、时间维等长逐通道互相关（两侧补 K//2 个零），最后输出 conv@out_weight。K 为正奇数，T>=1；每个通道只用自身卷积核。本题省略 Conformer 的归一化、Swish、dropout 和残差。",
    "function_name": "conformer_conv",
    "hint": "Transpose to (B,D,T), pad time, unfold windows, multiply each channel's K weights, then transpose back.",
    "hint_zh": "转成 (B,D,T)，在时间维补零、展开窗口，逐通道乘 K 个权重，再转回。",
    "tests": [
        {"name": "GLU and depthwise numeric reference", "code": """\nimport torch, torch.nn.functional as F\nx=torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]]); p=torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]])\ndw=torch.tensor([[1.,2.,3.],[-1.,0.,1.]]); o=torch.tensor([[1.,2.],[3.,4.]])\ny={fn}(x,p,dw,o); g=x*torch.sigmoid(x); r=torch.zeros_like(g)\nfor t in range(3):\n for c in range(2):\n  for j in range(3):\n   if 0<=t+j-1<3: r[0,t,c]+=g[0,t+j-1,c]*dw[c,j]\nassert y.shape==x.shape and torch.allclose(y,r@o,atol=1e-6)\n"""},
        {"name": "Singleton kernel and channel isolation", "code": """\nimport torch\nx=torch.tensor([[[2.,5.]]]); p=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]])\ndw=torch.tensor([[2.],[3.]]); o=torch.eye(2)\ny={fn}(x,p,dw,o)\nassert torch.allclose(y,torch.tensor([[[2.,7.5]]]))\nx2=x.clone(); x2[0,0,1]=100.\nassert torch.allclose({fn}(x2,p,dw,o)[...,0],y[...,0])\n"""},
        {"name": "Gradients", "code": """\nimport torch\nx=torch.randn(2,4,3,requires_grad=True); p=torch.randn(3,6,requires_grad=True); dw=torch.randn(3,3,requires_grad=True); o=torch.randn(3,3,requires_grad=True)\n{fn}(x,p,dw,o).square().sum().backward()\nassert all(z.grad is not None and torch.isfinite(z.grad).all() for z in (x,p,dw,o))\n"""},
    ],
    "solution": '''def conformer_conv(x, pointwise, depthwise, out_weight):
    import torch.nn.functional as F
    d, kernel = x.shape[-1], depthwise.shape[-1]
    z = x @ pointwise
    gated = z[..., :d] * torch.sigmoid(z[..., d:])
    channels = gated.transpose(1, 2)
    windows = F.pad(channels, (kernel // 2, kernel // 2)).unfold(-1, kernel, 1)
    convolved = (windows * depthwise[None, :, None, :]).sum(-1)
    return convolved.transpose(1, 2) @ out_weight''',
    "demo": """x=torch.randn(2,5,4)\nprint(conformer_conv(x,torch.randn(4,8),torch.randn(4,3),torch.eye(4)).shape)""",
}
