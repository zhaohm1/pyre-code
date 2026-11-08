"""Valid-frame mask for log-Mel feature batches."""
TASK = {
 "title":"ASR Valid-Frame Mask", "title_zh":"ASR 有效帧掩码", "difficulty":"Easy",
 "description_en":"Given padded log-Mel features `features` (B,T,F) and integer `lengths` (B,) with 0<=length<=T, return `(mask, masked)` where mask (B,T) is True exactly for t<length and masked (B,T,F) sets every padded frame to zero. Preserve valid zero-valued frames and gradients. This is an input-handling exercise; it does not extract features from waveforms.",
 "description_zh":"给定已填充的 log-Mel 特征 `features` (B,T,F) 与整数 `lengths` (B,)，0<=length<=T；返回 `(mask, masked)`。mask (B,T) 仅在 t<length 时为 True，masked (B,T,F) 将填充帧置零。保留有效帧中原有的零值及梯度。本题处理特征输入，不从波形提取特征。",
 "function_name":"asr_frame_mask", "hint":"Compare arange(T) with lengths[:,None], then expand the mask over F.", "hint_zh":"用 arange(T) 与 lengths[:,None] 比较，再沿 F 维广播。",
 "tests":[
  {"name":"Different lengths and padded values", "code":"""\nimport torch\nx=torch.tensor([[[1.,0.],[2.,3.],[99.,99.]],[[4.,5.],[88.,88.],[77.,77.]]]); m,y={fn}(x,torch.tensor([2,1]))\nassert m.dtype==torch.bool and m.tolist()==[[True,True,False],[True,False,False]]\nassert torch.equal(y,torch.tensor([[[1.,0.],[2.,3.],[0.,0.]],[[4.,5.],[0.,0.],[0.,0.]]]))\n"""},
  {"name":"Zero and full length plus gradients", "code":"""\nimport torch\nx=torch.randn(2,3,4,requires_grad=True); m,y={fn}(x,torch.tensor([0,3])); y.sum().backward()\nassert not m[0].any() and m[1].all() and torch.all(x.grad[0]==0) and torch.all(x.grad[1]==1)\n"""}],
 "solution":'''def asr_frame_mask(features, lengths):
    import torch
    mask = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
    return mask, features * mask.unsqueeze(-1).to(features.dtype)''',
 "demo":"""x=torch.randn(2,5,80); mask,valid=asr_frame_mask(x,torch.tensor([5,3])); print(mask,valid.shape)"""
}
