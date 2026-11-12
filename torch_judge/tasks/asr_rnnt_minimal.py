"""Minimal trainable RNN-T components and lattice loss."""

TASK = {
    "title":"Mini RNN-T", "title_zh":"最小 RNN-T", "difficulty":"Hard",
    "description_en":"Implement asr_rnnt_minimal(features, targets, embedding, w_encoder, w_e, w_p, bias, w_out, blank=0). Inputs are features (B,T,F), nonblank target IDs (B,U), embedding (V,Dp), w_encoder (F,De), w_e (De,H), w_p (Dp,H), bias (H), w_out (H,V); T>=1, U>=0. Encoder states are features@w_encoder. Prediction state u is the cumulative sum of embeddings of the first u target tokens, with zero state for u=0. The joint network computes tanh(encoder@w_e + predictor@w_p + bias)@w_out at every (t,u), returning logits (B,T,U+1,V). Return (logits, mean_nll), where mean_nll is the mean negative log probability of all legal RNN-T lattice paths per sample: blank moves (t,u) to (t+1,u), target[u] moves (t,u) to (t,u+1), and no emissions occur at t=T. Use logaddexp; this is a tiny full lattice, not optimized/pruned training.",
    "description_zh":"实现 asr_rnnt_minimal(features, targets, embedding, w_encoder, w_e, w_p, bias, w_out, blank=0)。输入依次是声学特征 (B,T,F)、无 blank 的目标 ID (B,U)、词嵌入 (V,Dp)、w_encoder (F,De)、w_e (De,H)、w_p (Dp,H)、bias (H)、w_out (H,V)；T>=1、U>=0。Encoder 状态为 features@w_encoder；Prediction 状态 u 是前 u 个目标 token 嵌入的累加，u=0 为零状态。Joint 在各 (t,u) 计算 tanh(encoder@w_e + predictor@w_p + bias)@w_out，返回 logits (B,T,U+1,V)。另返回每条样本所有合法 RNN-T 路径负对数概率的批平均：blank 从 (t,u) 到 (t+1,u)，target[u] 从 (t,u) 到 (t,u+1)，t=T 不再发射。用 logaddexp 合并路径。此题是小规模完整格点，不是优化或剪枝训练。",
    "function_name":"asr_rnnt_minimal",
    "hint":"The encoder reads acoustic frames; the prediction network summarizes only previously emitted nonblank tokens; the joint combines both axes. Construct U+1 predictor states by prefix summation, then broadcast encoder and predictor projections into (B,T,U+1,H). Apply log_softmax over V. In the forward lattice, process t then u in increasing order so same-frame token transitions are available; sum paths in log space and negate the terminal log probability.",
    "hint_zh":"Encoder 读取声学帧，Prediction Network 只汇总此前输出的非 blank token，Joint 融合两个维度。通过前缀累加构造 U+1 个预测状态，再把两路投影广播为 (B,T,U+1,H)，沿 V 维做 log_softmax。前向格点按 t、u 递增处理，使同帧 token 转移可继续传播；在对数域合并路径，终点对数概率取负得到 loss。",
    "tests": [
        {"name":"Uniform one-frame lattice and shape", "code":"""
import torch,math
f=torch.tensor([[[1.]]]); y=torch.tensor([[1]])
emb=torch.zeros(2,1); w=torch.zeros(1,1); b=torch.zeros(1); o=torch.zeros(1,2)
logits,loss={fn}(f,y,emb,w,w,w,b,o)
assert logits.shape==(1,1,2,2) and torch.allclose(logits,torch.zeros_like(logits))
assert abs(float(loss)-2*math.log(2))<1e-6
"""},
        {"name":"Empty target, batch and gradients", "code":"""
import torch,math
f=torch.randn(2,2,3,requires_grad=True); y=torch.empty(2,0,dtype=torch.long)
emb=torch.randn(4,5,requires_grad=True); we=torch.randn(3,6,requires_grad=True)
a=torch.randn(6,7,requires_grad=True); p=torch.randn(5,7,requires_grad=True)
b=torch.randn(7,requires_grad=True); o=torch.randn(7,4,requires_grad=True)
logits,loss={fn}(f,y,emb,we,a,p,b,o)
assert logits.shape==(2,2,1,4) and torch.isfinite(loss)
loss.backward(); assert all(z.grad is not None and torch.isfinite(z.grad).all() for z in [f,we,a,b,o])
"""},
    ],
    "solution":'''def asr_rnnt_minimal(features, targets, embedding, w_encoder, w_e, w_p, bias, w_out, blank=0):
    import torch
    encoder = features @ w_encoder
    target_emb = embedding[targets]
    zero = target_emb.new_zeros(target_emb.shape[0], 1, target_emb.shape[-1])
    predictor = torch.cat([zero, target_emb.cumsum(dim=1)], dim=1)
    hidden = torch.tanh((encoder @ w_e)[:, :, None, :] + (predictor @ w_p)[:, None, :, :] + bias)
    logits = hidden @ w_out
    log_probs = logits.log_softmax(dim=-1)
    batch, time, up, _ = logits.shape
    losses = []
    for i in range(batch):
        lattice = [[log_probs.new_tensor(float('-inf')) for _ in range(up)] for _ in range(time + 1)]
        lattice[0][0] = log_probs.new_tensor(0.)
        for t in range(time):
            for u in range(up):
                value = lattice[t][u]
                lattice[t + 1][u] = torch.logaddexp(lattice[t + 1][u], value + log_probs[i, t, u, blank])
                if u + 1 < up:
                    token = int(targets[i, u])
                    lattice[t][u + 1] = torch.logaddexp(lattice[t][u + 1], value + log_probs[i, t, u, token])
        losses.append(-lattice[time][-1])
    return logits, torch.stack(losses).mean()''',
    "demo":"""import torch
f=torch.randn(1,2,3); y=torch.tensor([[1]])
print(asr_rnnt_minimal(f,y,torch.randn(4,5),torch.randn(3,6),torch.randn(6,7),torch.randn(5,7),torch.zeros(7),torch.randn(7,4))[0].shape)""",
}
