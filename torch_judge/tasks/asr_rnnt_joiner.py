"""A small RNN-T joint network."""
TASK = {
 "title":"RNN-T Joint Network", "title_zh":"RNN-T 联合网络", "difficulty":"Medium",
 "description_en":"Given encoder states `encoder` (B,T,De), prediction states `predictor` (B,U+1,Dp), weights `w_e` (De,H), `w_p` (Dp,H), `bias` (H,), and `w_out` (H,V), return logits (B,T,U+1,V) = tanh(encoder@w_e + predictor@w_p + bias)@w_out with broadcasting over T and U. This is a toy joint network; recurrent prediction state construction and blank handling are separate.",
 "description_zh":"给定编码器状态 `encoder` (B,T,De)、预测状态 `predictor` (B,U+1,Dp)、权重 `w_e` (De,H)、`w_p` (Dp,H)、`bias` (H,) 和 `w_out` (H,V)，沿 T、U 广播计算 logits (B,T,U+1,V) = tanh(encoder@w_e + predictor@w_p + bias)@w_out。本题是简化联合网络；预测状态生成和 blank 处理在其他题中练习。",
 "function_name":"asr_rnnt_joiner", "hint":"Insert singleton dimensions after each projection, then broadcast before tanh.", "hint_zh":"两路投影后各增加一个单例维度，再广播相加、做 tanh。",
 "tests":[
  {"name":"Shape and independent numerical example", "code":"""\nimport torch\ne=torch.tensor([[[1.,2.],[3.,4.]]]); p=torch.tensor([[[5.,6.],[7.,8.],[9.,10.]]]); we=torch.eye(2); wp=torch.tensor([[2.,0.],[0.,3.]]); b=torch.tensor([1.,-1.]); o=torch.tensor([[1.,2.,3.],[4.,5.,6.]])\ny={fn}(e,p,we,wp,b,o); ref=torch.zeros(1,2,3,3)\nfor t in range(2):\n for u in range(3): ref[0,t,u]=torch.tanh(e[0,t]@we+p[0,u]@wp+b)@o\nassert y.shape==(1,2,3,3) and torch.allclose(y,ref,atol=1e-6)\n"""},
  {"name":"Batch independence and gradient", "code":"""\nimport torch\ne=torch.randn(2,3,4,requires_grad=True); p=torch.randn(2,2,5,requires_grad=True); we=torch.randn(4,6,requires_grad=True); wp=torch.randn(5,6,requires_grad=True); b=torch.randn(6,requires_grad=True); o=torch.randn(6,3,requires_grad=True)\ny={fn}(e,p,we,wp,b,o); y.square().sum().backward()\nassert y.shape==(2,3,2,3) and all(x.grad is not None for x in (e,p,we,wp,b,o))\n"""}],
 "solution":'''def asr_rnnt_joiner(encoder, predictor, w_e, w_p, bias, w_out):
    hidden = torch.tanh((encoder @ w_e)[:, :, None, :] + (predictor @ w_p)[:, None, :, :] + bias)
    return hidden @ w_out''',
 "demo":"""e=torch.randn(2,4,3); p=torch.randn(2,5,3)\nprint(asr_rnnt_joiner(e,p,torch.eye(3),torch.eye(3),torch.zeros(3),torch.randn(3,6)).shape)"""
}
