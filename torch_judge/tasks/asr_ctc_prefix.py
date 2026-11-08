"""Small-vocabulary CTC prefix beam search."""
TASK = {
 "title":"CTC Prefix Beam Search", "title_zh":"CTC 前缀束搜索", "difficulty":"Hard",
 "description_en":"For one utterance `log_probs` (T,V), `input_length` 1..T, blank=0 and positive `beam_size`, return up to beam_size `(prefix_tuple, log_probability)` pairs, sorted by descending total path probability then lexicographic prefix. Maintain separate probabilities for prefixes ending in blank and nonblank; merge paths that collapse to the same prefix. Prune to beam_size after each frame. This small-vocabulary exercise has no language model or streaming state.",
 "description_zh":"给定单条语音 `log_probs` (T,V)、有效长度 input_length（1..T）、blank=0 和正整数 beam_size，返回至多 beam_size 个 `(前缀元组, 总对数概率)`，按概率降序、同分按前缀字典序排列。分别保存以 blank 和非 blank 结束的概率，合并折叠为相同前缀的路径，每帧后保留 beam_size 个前缀。本题面向小词表，无语言模型或流式状态。",
 "function_name":"asr_ctc_prefix", "hint":"For a repeated token, a nonblank-ending path can stay on the same prefix; only a blank-ending path can extend it with the same token.", "hint_zh":"遇到相同 token 时，非 blank 结尾路径可留在原前缀；只有 blank 结尾路径可扩展出重复 token。",
 "tests":[
  {"name":"All paths merged without pruning", "code":"""\nimport torch,itertools,math\np=torch.tensor([[.2,.5,.3],[.4,.4,.2],[.3,.2,.5]]); got={fn}(p.log(),3,beam_size=27); ref={}\nfor path in itertools.product(range(3),repeat=3):\n out=[]; prev=None\n for token in path:\n  if token!=prev and token!=0: out.append(token)\n  prev=token\n key=tuple(out); ref[key]=ref.get(key,0.)+math.prod(p[t,v].item() for t,v in enumerate(path))\nassert [x[0] for x in got]==sorted(ref,key=lambda x:(-ref[x],x))\nassert all(abs(math.exp(score)-ref[prefix])<1e-6 for prefix,score in got)\n"""},
  {"name":"Beam limit, padding and repeated token", "code":"""\nimport torch,math\np=torch.tensor([[.1,.9],[.8,.2],[.1,.9],[.99,.01]])\na={fn}(p.log(),3,beam_size=2); b={fn}(p[:3].log(),3,beam_size=2)\nassert len(a)==2 and a==b and a[0][0]==(1,1)\n"""}],
 "solution":'''def asr_ctc_prefix(log_probs, input_length, beam_size, blank=0):
    import math
    beam = {(): (1.0, 0.0)}
    vocab = log_probs.shape[-1]
    for t in range(input_length):
        probs = log_probs[t].exp().tolist()
        next_beam = {}
        def add(prefix, blank_add=0.0, nonblank_add=0.0):
            pb, pn = next_beam.get(prefix, (0.0, 0.0))
            next_beam[prefix] = (pb + blank_add, pn + nonblank_add)
        for prefix, (pb, pn) in beam.items():
            add(prefix, blank_add=(pb + pn) * probs[blank])
            for token in range(vocab):
                if token == blank:
                    continue
                if prefix and token == prefix[-1]:
                    add(prefix, nonblank_add=pn * probs[token])
                    add(prefix + (token,), nonblank_add=pb * probs[token])
                else:
                    add(prefix + (token,), nonblank_add=(pb + pn) * probs[token])
        ranked = sorted((p for p in next_beam if sum(next_beam[p]) > 0), key=lambda p: (-(next_beam[p][0] + next_beam[p][1]), p))
        beam = {p: next_beam[p] for p in ranked[:beam_size]}
    return [(p, math.log(pb + pn)) for p, (pb, pn) in beam.items()]''',
 "demo":"""p=torch.tensor([[.1,.8,.1],[.4,.5,.1]]); print(asr_ctc_prefix(p.log(),2,3))"""
}
