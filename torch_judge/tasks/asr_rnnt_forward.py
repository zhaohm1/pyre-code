"""RNN-T lattice forward probability."""
TASK = {
 "title":"RNN-T Forward Probability", "title_zh":"RNN-T 前向概率", "difficulty":"Hard",
 "description_en":"Given one utterance `log_probs` (T,U+1,V), target IDs `target` of length U excluding blank, `input_length` 1..T, and blank=0, return the log probability of all monotone paths from lattice state (0,0) to (input_length,U). At (t,u), a blank emits no token and advances time to (t+1,u); if u<U, target[u] advances labels to (t,u+1). No emissions are possible after t=input_length. Use logaddexp. This is a small single-utterance forward exercise, not a complete optimized RNN-T loss.",
 "description_zh":"给定单条语音 `log_probs` (T,U+1,V)、长度 U 且不含 blank 的目标 ID `target`、有效帧数 input_length（1..T）、blank=0，返回格点 (0,0) 到 (input_length,U) 所有单调路径的总对数概率。在 (t,u) 发 blank 到 (t+1,u)；若 u<U，发 target[u] 到 (t,u+1)。t=input_length 后不可再发符号。使用 logaddexp。本题是单条语音的小格点前向递推，不是完整优化的 RNN-T 损失。",
 "function_name":"asr_rnnt_forward", "hint":"Maintain an (input_length+1)-by-(U+1) lattice; update blank and token transitions in increasing t,u order.", "hint_zh":"维护 (input_length+1)×(U+1) 格点，按递增的 t、u 更新 blank 与 token 转移。",
 "tests":[
  {"name":"Two independent paths to one token", "code":"""\nimport torch\np=torch.tensor([[[.7,.3],[.4,.6]],[[.2,.8],[.5,.5]]]); y={fn}(p.log(),[1],2)\nexpected=(p[0,0,1]*p[0,1,0]*p[1,1,0]+p[0,0,0]*p[1,0,1]*p[1,1,0]).log()\nassert torch.allclose(y,expected,atol=1e-6)\n"""},
  {"name":"Empty target and ignored suffix", "code":"""\nimport torch\np=torch.tensor([[[.8,.2]],[[.7,.3]],[[.1,.9]]]); y={fn}(p.log(),[],2)\nassert torch.allclose(y,(p[0,0,0]*p[1,0,0]).log())\nassert torch.allclose(y,{fn}(p[:2].log(),[],2))\n"""},
  {"name":"Differentiable lattice", "code":"""\nimport torch\nx=torch.randn(2,3,3,requires_grad=True); y={fn}(x.log_softmax(-1),[1,2],2); y.backward()\nassert x.grad is not None and torch.isfinite(x.grad).all()\n"""}],
 "solution":'''def asr_rnnt_forward(log_probs, target, input_length, blank=0):
    import torch
    u_max = len(target)
    lattice = [[log_probs.new_tensor(float('-inf')) for _ in range(u_max + 1)] for _ in range(input_length + 1)]
    lattice[0][0] = log_probs.new_tensor(0.0)
    for t in range(input_length):
        for u in range(u_max + 1):
            value = lattice[t][u]
            lattice[t + 1][u] = torch.logaddexp(lattice[t + 1][u], value + log_probs[t, u, blank])
            if u < u_max:
                lattice[t][u + 1] = torch.logaddexp(lattice[t][u + 1], value + log_probs[t, u, target[u]])
    return lattice[input_length][u_max]''',
 "demo":"""x=torch.randn(2,2,3).log_softmax(-1); print(asr_rnnt_forward(x,[1],2))"""
}
