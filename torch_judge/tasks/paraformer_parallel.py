"""Compact parallel decoder attention exercise."""

TASK = {
    "title": "Paraformer Parallel Token Prediction",
    "title_zh": "Paraformer 并行 token 预测",
    "difficulty": "Medium",
    "description_en": "Given all CIF token states `tokens` (B,U,D) and encoder states `encoder` (B,T,D), compute single-head noncausal token self-attention, then encoder cross-attention, then vocabulary logits (B,U,V). Signature: `paraformer_parallel(tokens, encoder, sq, sk, sv, cq, ck, cv, out)`. The first six weights have shape (D,D), `out` has shape (D,V). Self-attention uses `tokens` for Q/K/V; cross-attention uses the self-attention output for Q and `encoder` for K/V. Each attention score is divided by sqrt(D). Return all token logits in one call, without causal masking, recurrence, bias, residual, or final softmax. This is a focused exercise, not a complete Paraformer decoder.",
    "description_zh": "给定全部 CIF token 表示 `tokens` (B,U,D) 和编码器表示 `encoder` (B,T,D)，先计算单头、无因果遮蔽的 token 自注意力，再计算编码器交叉注意力，最后输出词表 logits (B,U,V)。签名：`paraformer_parallel(tokens, encoder, sq, sk, sv, cq, ck, cv, out)`。前六个权重形状为 (D,D)，`out` 为 (D,V)。自注意力以 `tokens` 生成 Q/K/V；交叉注意力以自注意力结果生成 Q，以 `encoder` 生成 K/V。每次注意力分数除以 sqrt(D)。一次返回所有 token 位置的 logits；不使用因果遮蔽、递归、偏置、残差或最终 softmax。这是聚焦练习，并非完整 Paraformer 解码器。",
    "function_name": "paraformer_parallel",
    "hint": "Self: softmax((tokens@sq) @ (tokens@sk).transpose(-1,-2) / sqrt(D)) @ (tokens@sv). Cross: use self output for Q, encoder for K/V. Then project with out.",
    "hint_zh": "自注意力使用 tokens 生成 Q/K/V；交叉注意力用自注意力结果生成 Q、encoder 生成 K/V；最后乘 out。",
    "tests": [
        {"name": "Shape and independent numerical reference", "code": """\nimport torch, math\ntorch.manual_seed(7)\nB,U,T,D,V=2,3,4,5,7\nx=torch.randn(B,U,D); e=torch.randn(B,T,D)\nw=[torch.randn(D,D) for _ in range(6)]+[torch.randn(D,V)]\ny={fn}(x,e,*w)\nq,k,v=(x@w[i] for i in range(3))\nh=torch.bmm(torch.softmax(torch.bmm(q,k.transpose(1,2))/math.sqrt(D),dim=-1),v)\nq=h@w[3]; k=e@w[4]; v=e@w[5]\nexpected=torch.bmm(torch.softmax(torch.bmm(q,k.transpose(1,2))/math.sqrt(D),dim=-1),v)@w[6]\nassert y.shape==(B,U,V) and torch.allclose(y,expected,atol=1e-5)\n"""},
        {"name": "Later token affects earlier output", "code": """\nimport torch\nI=torch.eye(2); x=torch.tensor([[[1.,0.],[0.,1.]]]); e=torch.tensor([[[1.,2.],[3.,4.]]])\ny1={fn}(x,e,I,I,I,I,I,I,I)\nx2=x.clone(); x2[:,1]=torch.tensor([3.,2.])\ny2={fn}(x2,e,I,I,I,I,I,I,I)\nassert not torch.allclose(y1[:,0],y2[:,0])\n"""},
        {"name": "Gradients for inputs and weights", "code": """\nimport torch\nx=torch.randn(1,3,4,requires_grad=True); e=torch.randn(1,5,4,requires_grad=True)\nw=[torch.randn(4,4,requires_grad=True) for _ in range(6)]+[torch.randn(4,6,requires_grad=True)]\n{fn}(x,e,*w).square().sum().backward()\nassert x.grad is not None and e.grad is not None and all(z.grad is not None for z in w)\n"""},
    ],
    "solution": '''def paraformer_parallel(tokens, encoder, sq, sk, sv, cq, ck, cv, out):
    d = tokens.shape[-1]
    self_scores = (tokens @ sq) @ (tokens @ sk).transpose(-1, -2) / (d ** 0.5)
    hidden = torch.softmax(self_scores, dim=-1) @ (tokens @ sv)
    cross_scores = (hidden @ cq) @ (encoder @ ck).transpose(-1, -2) / (d ** 0.5)
    context = torch.softmax(cross_scores, dim=-1) @ (encoder @ cv)
    return context @ out''',
    "demo": """x = torch.randn(2, 4, 8)
e = torch.randn(2, 12, 8)
w = [torch.randn(8, 8) for _ in range(6)] + [torch.randn(8, 20)]
print(paraformer_parallel(x, e, *w).shape)""",
}
