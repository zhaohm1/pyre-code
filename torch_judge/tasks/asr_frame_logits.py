"""Frame-level vocabulary projection."""
TASK = {
 "title":"ASR Frame-to-Vocabulary Projection", "title_zh":"ASR 帧级词表投影", "difficulty":"Easy",
 "description_en":"Given encoder states `states` (B,T,D), valid `lengths` (B,), `weight` (D,V), and `bias` (V,), return logits (B,T,V) = states@weight+bias on valid frames and exactly zero on padded frames. No softmax or token decoding. This isolates the output head of a toy ASR encoder; it does not define the encoder.",
 "description_zh":"给定编码器状态 `states` (B,T,D)、有效长度 `lengths` (B,)、`weight` (D,V) 和 `bias` (V,)，返回 logits (B,T,V)：有效帧为 states@weight+bias，填充帧严格为零。不做 softmax 或 token 解码。本题只练习简化 ASR 编码器的输出层，不定义编码器。",
 "function_name":"asr_frame_logits", "hint":"Project first, then zero positions with t>=length so bias cannot leak into padding.", "hint_zh":"先投影，再将 t>=length 的位置清零，避免偏置留在填充帧。",
 "tests":[
  {"name":"Numerical projection and bias-free padding", "code":"""\nimport torch\nx=torch.tensor([[[1.,2.],[9.,9.]],[[3.,4.],[5.,6.]]]); w=torch.tensor([[1.,2.],[3.,4.]]); b=torch.tensor([5.,7.]); y={fn}(x,torch.tensor([1,2]),w,b)\nr=x@w+b; r[0,1]=0\nassert y.shape==(2,2,2) and torch.equal(y,r)\n"""},
  {"name":"Zero length and gradients", "code":"""\nimport torch\nx=torch.randn(2,3,4,requires_grad=True); w=torch.randn(4,5,requires_grad=True); b=torch.randn(5,requires_grad=True)\ny={fn}(x,torch.tensor([0,2]),w,b); y.sum().backward()\nassert torch.all(y[0]==0) and torch.all(y[1,2]==0) and torch.all(x.grad[0]==0) and w.grad is not None and b.grad is not None\n"""}],
 "solution":'''def asr_frame_logits(states, lengths, weight, bias):
    import torch
    logits = states @ weight + bias
    valid = torch.arange(states.shape[1], device=states.device)[None, :] < lengths[:, None]
    return logits * valid.unsqueeze(-1).to(logits.dtype)''',
 "demo":"""s=torch.randn(2,4,8); print(asr_frame_logits(s,torch.tensor([4,2]),torch.randn(8,6),torch.zeros(6)).shape)"""
}
