"""Simplified relative-position self-attention exercise."""

TASK = {
    "title": "Conformer Relative-Position Attention",
    "title_zh": "Conformer 相对位置注意力",
    "difficulty": "Medium",
    "description_en": "Compute one-head attention with relative-position score bias. `q,k,v` are (B,T,D), and `rel_bias` is (2T-1,). Score(i,j) = dot(q_i,k_j)/sqrt(D) + rel_bias[j-i+T-1]. Apply softmax over keys and return (B,T,D). This isolates relative offsets; it omits Conformer's learned positional projections, heads, masks and output projection. T must be at least 1.",
    "description_zh": "计算单头、带相对位置分数偏置的自注意力。`q,k,v` 为 (B,T,D)，`rel_bias` 为 (2T-1,)。分数(i,j) = dot(q_i,k_j)/sqrt(D) + rel_bias[j-i+T-1]。在 key 维做 softmax，返回 (B,T,D)。本题只练习相对位移；省略 Conformer 的可学习位置投影、多头、掩码和输出投影。T 至少为 1。",
    "function_name": "conformer_relative_attention",
    "hint": "Build the T-by-T table of j-i+T-1 indices, then add the selected bias before softmax.",
    "hint_zh": "构造 j-i+T-1 的 T×T 索引表，在 softmax 前加入对应偏置。",
    "tests": [
        {"name": "Relative offsets and numerical result", "code": """\nimport torch, math\nq=torch.zeros(1,3,2); k=q.clone(); v=torch.tensor([[[1.,0.],[0.,2.],[3.,0.]]])\nb=torch.tensor([-2.,-1.,0.,1.,2.])\ny={fn}(q,k,v,b)\nr=torch.stack([torch.softmax(torch.tensor([b[j-i+2] for j in range(3)]),0)@v[0] for i in range(3)])\nassert y.shape==(1,3,2) and torch.allclose(y[0],r,atol=1e-6)\n"""},
        {"name": "Content score, batches and singleton", "code": """\nimport torch, math\nq=torch.tensor([[[1.,0.],[0.,1.]],[[2.,0.],[0.,2.]]]); k=q.clone(); v=torch.arange(8.).reshape(2,2,2); b=torch.zeros(3)\ny={fn}(q,k,v,b)\nr=torch.softmax(q@k.transpose(-2,-1)/math.sqrt(2),-1)@v\nassert torch.allclose(y,r,atol=1e-6) and not torch.allclose(y[0],y[1])\nassert torch.allclose({fn}(q[:,:1],k[:,:1],v[:,:1],torch.zeros(1)),v[:,:1])\n"""},
        {"name": "Gradients", "code": """\nimport torch\nq=torch.randn(1,3,2,requires_grad=True); k=torch.randn(1,3,2,requires_grad=True); v=torch.randn(1,3,2,requires_grad=True); b=torch.randn(5,requires_grad=True)\n{fn}(q,k,v,b).square().sum().backward()\nassert all(z.grad is not None and torch.isfinite(z.grad).all() for z in (q,k,v,b))\n"""},
    ],
    "solution": '''def conformer_relative_attention(q, k, v, rel_bias):
    import math
    t, d = q.shape[1], q.shape[2]
    positions = torch.arange(t, device=q.device)
    offsets = positions[None, :] - positions[:, None] + t - 1
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    return torch.softmax(scores + rel_bias[offsets], dim=-1) @ v''',
    "demo": """q=torch.randn(1,4,8); k=torch.randn(1,4,8); v=torch.randn(1,4,8)\nprint(conformer_relative_attention(q,k,v,torch.zeros(7)).shape)""",
}
