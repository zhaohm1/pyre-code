"""CTC log-domain forward probability."""
TASK = {
 "title":"CTC Forward Probability", "title_zh":"CTC 前向概率", "difficulty":"Hard",
 "description_en":"`log_probs` is (T,V) log probabilities, `target` is a sequence of nonblank IDs, `input_length` is 1..T, and `blank=0`. Return the scalar log probability of all CTC paths over the first input_length frames that collapse to target. Use log-domain recurrence on blank-interleaved target; a skip transition is allowed only when the destination token is nonblank and differs from the token two states earlier. Return -inf if impossible. This is one utterance, not a batched training loss.",
 "description_zh":"`log_probs` 是 (T,V) 对数概率，`target` 为不含 blank 的 ID 序列，`input_length` 为 1..T，blank 默认为 0。返回前 input_length 帧中所有折叠成 target 的 CTC 路径总对数概率。对插入 blank 的目标做对数域递推；仅当目的状态非 blank 且不同于前两个状态的 token 时允许跳转。无可行路径返回 -inf。本题是单条语音的概率计算，不是批量训练损失。",
 "function_name":"asr_ctc_forward", "hint":"Initialize states 0 and 1 at frame 0; each later state combines stay, previous, and permitted skip with logsumexp.", "hint_zh":"首帧初始化状态 0、1；以后每个状态用 logsumexp 合并停留、前一步和允许的跨步。",
 "tests":[
  {"name":"Enumerated paths and padded suffix", "code":"""\nimport torch,itertools,math\np=torch.tensor([[0.4,0.6],[0.5,0.5],[0.2,0.8]]); lp=p.log(); target=(1,)\nways=[]\nfor path in itertools.product(range(2),repeat=2):\n out=[]; prev=None\n for v in path:\n  if v!=prev and v!=0: out.append(v)\n  prev=v\n if tuple(out)==target: ways.append(p[0,path[0]]*p[1,path[1]])\nexpected=torch.stack(ways).sum().log(); y={fn}(lp,target,2)\nassert torch.allclose(y,expected,atol=1e-6)\nassert torch.allclose(y,{fn}(torch.cat([lp[:2],torch.tensor([[100.,-100.]])]),target,2))\n"""},
  {"name":"Repeated target requires intervening blank; empty target", "code":"""\nimport torch,math\np=torch.tensor([[0.2,0.8],[0.3,0.7],[0.4,0.6]])\nassert torch.isneginf({fn}(p.log(),[1,1],2))\nassert torch.allclose({fn}(p.log(),[1,1],3),(p[0,1]*p[1,0]*p[2,1]).log())\nassert torch.allclose({fn}(p.log(),[],3),p[:,0].log().sum())\n"""},
  {"name":"Differentiable finite path", "code":"""\nimport torch\nx=torch.randn(3,3,requires_grad=True); y={fn}(x.log_softmax(-1),[1,2],3); y.backward()\nassert x.grad is not None and torch.isfinite(x.grad).all()\n"""}],
 "solution":'''def asr_ctc_forward(log_probs, target, input_length, blank=0):
    import torch
    ext = [blank]
    for token in target:
        ext.extend((token, blank))
    neg = log_probs.new_tensor(float('-inf'))
    states = [neg for _ in ext]
    states[0] = log_probs[0, blank]
    if len(ext) > 1:
        states[1] = log_probs[0, ext[1]]
    for t in range(1, input_length):
        next_states = []
        for s, token in enumerate(ext):
            incoming = [states[s]]
            if s > 0:
                incoming.append(states[s - 1])
            if s > 1 and token != blank and token != ext[s - 2]:
                incoming.append(states[s - 2])
            next_states.append(torch.logsumexp(torch.stack(incoming), 0) + log_probs[t, token])
        states = next_states
    return torch.logsumexp(torch.stack(states[-2:] if len(states)>1 else states), 0)''',
 "demo":"""p=torch.tensor([[.3,.7],[.6,.4]]); print(asr_ctc_forward(p.log(),[1],2))"""
}
