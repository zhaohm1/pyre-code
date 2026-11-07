"""Continuous integrate-and-fire alignment exercise."""

TASK = {
    "title": "CIF Integrate and Fire",
    "title_zh": "CIF 积分与触发",
    "difficulty": "Medium",
    "description_en": "Implement threshold-1 CIF over `frames` (B,T,D) and nonnegative `alpha` (B,T). Consume frames in order. Each full unit of accumulated weight fires one token embedding, the weighted sum of frame vectors contributing to that unit. Split a frame's weight across a boundary; one frame may fire several tokens. Return `(tokens, lengths, tail, tail_weight)` where tokens is zero-padded to (B,max(lengths),D), lengths is (B,) integer, and the unfinished weighted sum and weight are (B,D) and (B,). An exact threshold produces an empty tail. No tail token is emitted. Preserve gradients through token values.",
    "description_zh": "对 `frames` (B,T,D) 和非负 `alpha` (B,T) 实现阈值为 1 的 CIF。按帧顺序累积权重；每满 1 即触发一个 token 表示，其值为当前单位内帧向量的加权和。跨越阈值的帧须拆分权重；单帧也可能触发多个 token。返回 `(tokens, lengths, tail, tail_weight)`：tokens 补零至 (B,max(lengths),D)，lengths 为 (B,) 整数，未完成部分的加权和与权重分别为 (B,D)、(B,)。恰好触发时尾部为空；不额外输出尾部 token。token 值须保留梯度。",
    "function_name": "cif_integrate",
    "hint": "For each frame, repeatedly consume min(remaining frame weight, 1-current mass). Fire at mass 1, then reset the accumulator. Pad emitted lists only after all batch rows finish.",
    "hint_zh": "逐帧反复消耗 min(当前帧剩余权重, 1-当前累积量)；满 1 后输出并清空累积器。所有样本完成后再对输出补零。",
    "tests": [
        {"name": "Exact boundary and split frame", "code": """\nimport torch\nx=torch.tensor([[[2.],[4.],[8.]]])\na=torch.tensor([[0.5,1.0,0.5]])\ny,n,t,w={fn}(x,a)\nassert n.tolist()==[2] and y.shape==(1,2,1)\nassert torch.allclose(y[0,:,0],torch.tensor([3.,6.]))\nassert torch.allclose(t,torch.zeros_like(t)) and torch.allclose(w,torch.zeros_like(w))\n"""},
        {"name": "One frame crosses multiple boundaries", "code": """\nimport torch\ny,n,t,w={fn}(torch.tensor([[[3.]]]),torch.tensor([[2.25]]))\nassert n.tolist()==[2] and torch.allclose(y[0,:,0],torch.tensor([3.,3.]))\nassert torch.allclose(t,torch.tensor([[0.75]])) and torch.allclose(w,torch.tensor([0.25]))\n"""},
        {"name": "Batch padding and unfinished tail", "code": """\nimport torch\nx=torch.tensor([[[2.],[4.]],[[5.],[9.]]])\na=torch.tensor([[0.5,0.75],[0.2,0.3]])\ny,n,t,w={fn}(x,a)\nassert n.tolist()==[1,0] and y.shape==(2,1,1)\nassert torch.allclose(y[:,0,0],torch.tensor([3.,0.]))\nassert torch.allclose(t[:,0],torch.tensor([1.,3.7]))\nassert torch.allclose(w,torch.tensor([0.25,0.5]))\n"""},
        {"name": "Gradient through fired embedding", "code": """\nimport torch\nx=torch.tensor([[[2.],[4.]]],requires_grad=True)\ny,n,t,w={fn}(x,torch.tensor([[0.5,0.5]]))\ny.sum().backward()\nassert torch.allclose(x.grad[:,:,0],torch.tensor([[0.5,0.5]]))\n"""},
    ],
    "solution": '''def cif_integrate(frames, alpha):
    if torch.any(alpha < 0):
        raise ValueError("alpha must be nonnegative")
    B, T, D = frames.shape
    rows, tails, masses, lengths = [], [], [], []
    for b in range(B):
        emitted = []
        acc = frames.new_zeros(D)
        mass = 0.0
        for i in range(T):
            remaining = float(alpha[b, i].item())
            while remaining > 1e-7:
                take = min(remaining, 1.0 - mass)
                acc = acc + frames[b, i] * take
                mass += take
                remaining -= take
                if mass >= 1.0 - 1e-7:
                    emitted.append(acc)
                    acc = frames.new_zeros(D)
                    mass = 0.0
        rows.append(emitted)
        tails.append(acc)
        masses.append(mass)
        lengths.append(len(emitted))
    max_len = max(lengths, default=0)
    padded = [torch.stack(row + [frames.new_zeros(D)] * (max_len - len(row))) for row in rows] if max_len else []
    tokens = torch.stack(padded) if padded else frames.new_zeros(B, 0, D)
    return tokens, torch.tensor(lengths, device=frames.device), torch.stack(tails), alpha.new_tensor(masses)''',
    "demo": """x = torch.tensor([[[2.], [4.], [8.]]])
print(cif_integrate(x, torch.tensor([[0.5, 1.0, 0.5]])))""",
}
