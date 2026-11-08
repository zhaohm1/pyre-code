"""CTC greedy frame decoding."""
TASK = {
 "title":"CTC Greedy Decode", "title_zh":"CTC 贪心解码", "difficulty":"Easy",
 "description_en":"For frame logits `logits` (B,T,V), valid `lengths` (B,), and blank=0, choose argmax per valid frame (lowest ID wins ties), then CTC-collapse repeats before removing blanks. Return a list of token-ID tuples, one per batch item. Ignore padded frames even if their logits are very large. This is framewise greedy decoding, not beam search.",
 "description_zh":"给定帧级 logits (B,T,V)、有效长度 `lengths` (B,) 与 blank=0，对每个有效帧取 argmax（同分选最小 ID），再按 CTC 规则先合并连续重复、后删除 blank。返回每条语音一个 token ID 元组组成的列表。即使填充帧 logits 很大也必须忽略。本题是逐帧贪心解码，不是束搜索。",
 "function_name":"asr_ctc_greedy", "hint":"Slice each row at its valid length before collapse; do not remove blanks before merging repeats.", "hint_zh":"每行先裁到有效长度再折叠；不能先删 blank 再合并重复。",
 "tests":[
  {"name":"Blank-separated repeats and padding", "code":"""\nimport torch\nids=torch.tensor([[1,1,0,1,2],[2,2,2,1,1]]); z=torch.nn.functional.one_hot(ids,3).float()*10; z[1,2:,0]=100\nassert {fn}(z,torch.tensor([5,2]))==[(1,1,2),(2,)]\n"""},
  {"name":"Empty, ties and custom blank", "code":"""\nimport torch\nz=torch.tensor([[[1.,1.,0.],[0.,0.,5.]]]); assert {fn}(z,torch.tensor([0]))==[()]\nassert {fn}(z,torch.tensor([2]),blank=2)==[(0,)]\n"""}],
 "solution":'''def asr_ctc_greedy(logits, lengths, blank=0):
    results = []
    for row, length in zip(logits, lengths):
        previous = None
        tokens = []
        for token in row[:int(length)].argmax(-1).tolist():
            if token != previous and token != blank:
                tokens.append(token)
            previous = token
        results.append(tuple(tokens))
    return results''',
 "demo":"""x=torch.randn(2,5,4); print(asr_ctc_greedy(x,torch.tensor([5,3])))"""
}
