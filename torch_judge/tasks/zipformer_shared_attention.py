"""Shared attention-weight exercise inspired by Zipformer."""

TASK = {
    "title": "Zipformer Shared Attention Weights",
    "title_zh": "Zipformer 注意力权重共享",
    "difficulty": "Medium",
    "description_en": "Given q,k (B,T,Dk) and two distinct value tensors v1 (B,T,D1), v2 (B,T,D2), compute one attention matrix softmax(q@k^T/sqrt(Dk), key axis), then return `(weights, weights@v1, weights@v2)`. T>=1. The same weights must aggregate both value streams. This isolates the sharing idea; it omits Zipformer's positional scores, nonlinear attention, extra projections, masks and full block.",
    "description_zh": "给定 q,k (B,T,Dk) 和两路值 v1 (B,T,D1)、v2 (B,T,D2)，只计算一组权重 softmax(q@k^T/sqrt(Dk)，沿 key 维)，返回 `(weights, weights@v1, weights@v2)`。T>=1；两路值必须复用同一组权重。本题只练习共享思想，省略 Zipformer 的位置分数、非线性注意力、其他投影、掩码和完整块。",
    "function_name": "zipformer_shared_attention",
    "hint": "Compute and normalize scores once, then apply the resulting matrix to each value tensor.",
    "hint_zh": "只计算和归一化一次分数，再分别与两路值相乘。",
    "tests": [
        {"name": "Independent numerical reference and different value dimensions", "code": """\nimport torch, math\nq=torch.tensor([[[1.,0.],[0.,1.]]]); k=q.clone(); v1=torch.tensor([[[1.],[3.]]]); v2=torch.tensor([[[1.,2.,3.],[4.,5.,6.]]])\nw,a,b={fn}(q,k,v1,v2); ref=torch.softmax(q@k.transpose(-2,-1)/math.sqrt(2),-1)\nassert w.shape==(1,2,2) and a.shape==(1,2,1) and b.shape==(1,2,3)\nassert torch.allclose(w,ref) and torch.allclose(a,ref@v1) and torch.allclose(b,ref@v2)\n"""},
        {"name": "Value independence and singleton", "code": """\nimport torch\nq=torch.randn(2,3,4); k=torch.randn(2,3,4); a=torch.randn(2,3,2); b=torch.randn(2,3,5)\nw,y,z={fn}(q,k,a,b); w2,y2,z2={fn}(q,k,a,b+10)\nassert torch.allclose(w,w2) and torch.allclose(y,y2) and torch.allclose(z2-z,torch.full_like(z,10.))\nw,y,z={fn}(q[:,:1],k[:,:1],a[:,:1],b[:,:1])\nassert torch.allclose(w,torch.ones_like(w)) and torch.allclose(y,a[:,:1]) and torch.allclose(z,b[:,:1])\n"""},
        {"name": "Gradients to both streams", "code": """\nimport torch\nq=torch.randn(1,3,2,requires_grad=True); k=torch.randn(1,3,2,requires_grad=True); a=torch.randn(1,3,2,requires_grad=True); b=torch.randn(1,3,4,requires_grad=True)\n_,y,z={fn}(q,k,a,b); (y.square().sum()+z.square().sum()).backward()\nassert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q,k,a,b))\n"""},
    ],
    "solution": '''def zipformer_shared_attention(q, k, v1, v2):
    import math
    weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
    return weights, weights @ v1, weights @ v2''',
    "demo": """q=torch.randn(1,3,4); k=torch.randn(1,3,4)\nprint([z.shape for z in zipformer_shared_attention(q,k,torch.randn(1,3,2),torch.randn(1,3,5))])""",
}
