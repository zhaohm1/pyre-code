"""CTC candidate attention rescoring."""
TASK = {
 "title":"ASR Attention Rescoring", "title_zh":"ASR 注意力重打分", "difficulty":"Medium",
 "description_en":"Given nonempty `candidates` as `(token_tuple, ctc_log_score)` pairs, callable `attention_score(token_tuple)` and 0<=ctc_weight<=1, compute combined score = ctc_weight*CTC + (1-ctc_weight)*attention. Return `(best_token_tuple, ranked_pairs)` where ranked_pairs contains `(token_tuple, combined_score)` sorted by descending score, then lexicographic token tuple on ties. Do not mutate candidates. This models score fusion of a CTC-first, attention-second decoder; attention scores are supplied, not learned here.",
 "description_zh":"给定非空 CTC 候选列表，每项为 `(token 元组, CTC 对数分数)`，可调用的 `attention_score(token 元组)` 与 0<=ctc_weight<=1，计算融合分数 = ctc_weight*CTC + (1-ctc_weight)*注意力分数。返回 `(最佳 token 元组, ranked_pairs)`；后者包含 `(token 元组, 融合分数)`，按分数降序、同分按 token 元组字典序排序。不修改原候选。本题模拟 CTC 一遍候选、注意力二遍重排；注意力分数由题目提供。",
 "function_name":"asr_attention_rescore", "hint":"Score every candidate once; use the tuple as a deterministic secondary sort key.", "hint_zh":"每个候选评分一次；用 token 元组作确定性的第二排序键。",
 "tests":[
  {"name":"Attention changes winner and scores", "code":"""\nc=[((1,),-0.1),((2,),-0.4)]; a=lambda p:0. if p==(1,) else 2.\nbest,ranked={fn}(c,a,0.5)\nassert best==(2,) and ranked==[((2,),0.8),((1,),-0.05)] and c==[((1,),-0.1),((2,),-0.4)]\n"""},
  {"name":"Tie and empty candidate rejection", "code":"""\nbest,ranked={fn}([((2,),0.),((1,),0.)],lambda p:0.,0.5)\nassert best==(1,) and [x[0] for x in ranked]==[(1,),(2,)]\ntry: {fn}([],lambda p:0.,0.5)\nexcept ValueError: pass\nelse: raise AssertionError('empty candidate list must fail')\n"""}],
 "solution":'''def asr_attention_rescore(candidates, attention_score, ctc_weight):
    if not candidates:
        raise ValueError('at least one CTC candidate is required')
    ranked = [(tuple(tokens), ctc_weight * float(ctc) + (1 - ctc_weight) * float(attention_score(tuple(tokens)))) for tokens, ctc in candidates]
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[0][0], ranked''',
 "demo":"""print(asr_attention_rescore([((1,),-1.),((2,),-2.)],lambda p:1. if p==(2,) else 0.,0.3))"""
}
