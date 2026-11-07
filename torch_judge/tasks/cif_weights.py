"""CIF frame weights and target-length normalization."""

TASK = {
    "title": "CIF Weights and Length",
    "title_zh": "CIF 权重与长度",
    "difficulty": "Easy",
    "description_en": "Prepare frame weights for a CIF predictor. `alpha` has shape (B,T), `valid` is a boolean mask of the same shape, and `target_lengths` has shape (B,). Return `(raw_lengths, scaled_alpha)`: the sum of valid original weights (B,), and weights normalized so each row sums to its target length. Invalid frames remain zero. If a row has zero valid weight, its target must be zero and its scaled weights remain zero. Do not round the raw lengths. Preserve gradients through nonzero weights.",
    "description_zh": "为 CIF 预测器准备帧权重。`alpha` 形状为 (B,T)，`valid` 是同形状的布尔有效帧掩码，`target_lengths` 形状为 (B,)。返回 `(raw_lengths, scaled_alpha)`：有效原始权重之和 (B,)，以及每行权重和等于目标 token 数的归一化权重。无效帧始终为零。若某行有效权重和为零，则其目标长度必须为零，该行归一化权重保持零。原始长度不取整；非零权重的计算须保留梯度。",
    "function_name": "cif_weights",
    "hint": "Mask first, sum per row, then scale each nonzero row by target_lengths/raw_lengths. Reject a positive target for a zero-sum row.",
    "hint_zh": "先遮蔽再逐行求和；非零行乘以 target_lengths/raw_lengths。零权重行若目标长度为正则报错。",
    "tests": [
        {"name": "Mask and independent sums", "code": """\nimport torch\na = torch.tensor([[0.2,0.3,9.0],[0.4,0.6,8.0]])\nm = torch.tensor([[True,True,False],[True,True,False]])\nraw, scaled = {fn}(a,m,torch.tensor([1,2]))\nassert torch.allclose(raw,torch.tensor([0.5,1.0]))\nassert torch.allclose(scaled,torch.tensor([[0.4,0.6,0.0],[0.8,1.2,0.0]]))\n"""},
        {"name": "Zero row and invalid frame invariance", "code": """\nimport torch\na = torch.tensor([[0.,0.,100.],[0.2,0.8,500.]])\nm = torch.tensor([[True,True,False],[True,True,False]])\nr,s = {fn}(a,m,torch.tensor([0,1]))\nassert torch.allclose(r,torch.tensor([0.,1.])) and torch.allclose(s.sum(1),torch.tensor([0.,1.]))\na[:,2] = -900\nr2,s2 = {fn}(a,m,torch.tensor([0,1]))\nassert torch.allclose(r,r2) and torch.allclose(s,s2)\n"""},
        {"name": "Impossible positive target", "code": """\nimport torch\ntry:\n    {fn}(torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool),torch.tensor([1]))\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Expected ValueError for zero weight and positive target')\n"""},
        {"name": "Gradient through normalization", "code": """\nimport torch\na = torch.tensor([[0.2,0.8]],requires_grad=True)\nr,s = {fn}(a,torch.ones(1,2,dtype=torch.bool),torch.tensor([1]))\n(s[0,0]+r[0]).backward()\nassert a.grad is not None and torch.isfinite(a.grad).all()\n"""},
    ],
    "solution": '''def cif_weights(alpha, valid, target_lengths):
    masked = alpha * valid.to(alpha.dtype)
    raw = masked.sum(dim=1)
    targets = target_lengths.to(alpha.dtype)
    if torch.any((raw == 0) & (targets > 0)):
        raise ValueError("positive target length with zero valid weight")
    scale = torch.where(raw > 0, targets / raw.clamp_min(1e-12), torch.zeros_like(raw))
    return raw, masked * scale[:, None]''',
    "demo": """alpha = torch.tensor([[0.2, 0.3, 0.0]])
valid = torch.tensor([[True, True, False]])
print(cif_weights(alpha, valid, torch.tensor([1])))""",
}
