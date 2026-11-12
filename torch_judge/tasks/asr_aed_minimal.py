"""Minimal teacher-forced attention encoder-decoder."""

TASK = {
    "title": "Mini AED Teacher Forcing", "title_zh": "最小 AED 与教师强制", "difficulty": "Hard",
    "description_en": "Implement asr_aed_minimal(encoder, decoder_inputs, targets, embedding, w_q, w_k, w_v, w_out, pad_id=-1). Shapes: encoder (B,T,D), decoder_inputs and targets (B,U), embedding (V,D), attention weights (D,D), output weight (D,V). Compute causal scaled dot-product self-attention on decoder token embeddings, then cross-attention from decoder states to all encoder frames, then logits (B,U,V). Return (logits, mean_cross_entropy, cross_weights), with cross_weights (B,U,T). Ignore pad_id in the loss. Decoder inputs are BOS followed by the previous ground-truth tokens; T and U are positive and at least one target is unpadded. This omits residual blocks and a learned encoder.",
    "description_zh": "实现 asr_aed_minimal(encoder, decoder_inputs, targets, embedding, w_q, w_k, w_v, w_out, pad_id=-1)。形状依次为：encoder (B,T,D)，解码输入与目标 (B,U)，词嵌入 (V,D)，注意力权重 (D,D)，输出权重 (D,V)。对解码 token 嵌入做因果缩放点积自注意力，再以解码状态查询全部 encoder 帧做交叉注意力，输出 logits (B,U,V)。返回 (logits, 平均交叉熵, cross_weights)，其中权重形状 (B,U,T)；loss 忽略 pad_id。解码输入为 BOS 加前一个真实 token；T、U 均为正且至少有一个非填充目标。此简化模型不含残差块和可训练 encoder。",
    "function_name": "asr_aed_minimal",
    "hint": "AED uses token history and acoustic frames to predict the next token. Teacher forcing shifts the gold sequence into decoder_inputs. In self-attention, hide key positions later than each query before softmax; cross-attention has no frame-causal mask. Scale by sqrt(D), normalize across keys, and exclude padded targets from the mean loss.",
    "hint_zh": "AED 根据历史 token 和声学帧预测下一 token；教师强制将真实序列右移后作为 decoder_inputs。自注意力在 softmax 前遮蔽晚于当前 query 的 key；交叉注意力不对声学帧做因果遮蔽。点积除以 sqrt(D)，沿 key 维归一化，并从平均 loss 中排除填充目标。",
    "tests": [
        {"name": "Shape, normalization and gradients", "code": """
import torch
B,T,U,D,V=2,3,4,5,7
e=torch.randn(B,T,D,requires_grad=True); emb=torch.randn(V,D,requires_grad=True)
w=[torch.randn(D,D,requires_grad=True) for _ in range(3)]; o=torch.randn(D,V,requires_grad=True)
x=torch.tensor([[0,1,2,3],[0,2,1,0]]); y=torch.tensor([[1,2,3,4],[2,1,0,-1]])
logits,loss,a={fn}(e,x,y,emb,*w,o)
assert logits.shape==(B,U,V) and a.shape==(B,U,T)
assert torch.allclose(a.sum(-1),torch.ones(B,U),atol=1e-6)
loss.backward()
assert all(z.grad is not None and torch.isfinite(z.grad).all() for z in [e,emb,*w,o])
"""},
        {"name": "Causal tokens and global acoustic attention", "code": """
import torch
D=2; e=torch.tensor([[[1.,0.],[0.,1.]]]); emb=torch.tensor([[1.,0.],[0.,1.],[2.,1.]])
eye=torch.eye(D); out=torch.tensor([[1.,0.,-1.],[0.,1.,-1.]])
x=torch.tensor([[0,1]]); y=torch.tensor([[1,2]])
a={fn}(e,x,y,emb,eye,eye,eye,out)[0]
b={fn}(e,torch.tensor([[0,2]]),y,emb,eye,eye,eye,out)[0]
assert torch.allclose(a[:,0],b[:,0],atol=1e-6)
e2=e.clone(); e2[:,1]+=5
c={fn}(e2,x,y,emb,eye,eye,eye,out)[0]
assert not torch.allclose(a[:,0],c[:,0])
"""},
        {"name": "Padded target excluded from loss", "code": """
import torch,torch.nn.functional as F
e=torch.randn(1,2,3); emb=torch.randn(5,3); w=torch.eye(3); o=torch.randn(3,5)
x=torch.tensor([[0,1,2]]); y=torch.tensor([[1,2,-1]])
logits,loss,_={fn}(e,x,y,emb,w,w,w,o)
expected=F.cross_entropy(logits[:,:2].reshape(-1,5),y[:,:2].reshape(-1))
assert torch.allclose(loss,expected,atol=1e-6)
"""},
    ],
    "solution": '''def asr_aed_minimal(encoder, decoder_inputs, targets, embedding, w_q, w_k, w_v, w_out, pad_id=-1):
    import math
    import torch
    import torch.nn.functional as F
    x = embedding[decoder_inputs]
    d = x.shape[-1]
    scores = x @ x.transpose(-2, -1) / math.sqrt(d)
    causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
    decoder = torch.softmax(scores.masked_fill(causal, float('-inf')), dim=-1) @ x
    q, k, v = decoder @ w_q, encoder @ w_k, encoder @ w_v
    cross_weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)
    logits = (cross_weights @ v) @ w_out
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=pad_id)
    return logits, loss, cross_weights''',
    "demo": """import torch
e=torch.randn(1,3,4); emb=torch.randn(6,4); w=torch.eye(4); o=torch.randn(4,6)
print(asr_aed_minimal(e,torch.tensor([[0,1]]),torch.tensor([[1,2]]),emb,w,w,w,o)[0].shape)""",
}
