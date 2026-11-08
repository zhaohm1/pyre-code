"""Bounded RNN-T greedy decoding."""
TASK = {
 "title":"RNN-T Greedy Decode", "title_zh":"RNN-T 贪心解码", "difficulty":"Medium",
 "description_en":"`score(t,prefix)` returns V logits for frame t and emitted token tuple prefix. With `input_length` frames, blank=0 and positive `max_output_tokens`, repeatedly argmax (lowest ID wins ties): blank advances t; nonblank appends token without advancing t. Stop at t=input_length or immediately when max_output_tokens is reached. Return `(tokens_tuple, truncated_bool)`; truncated is True only when the token limit stops decoding. Repeated tokens remain repeated. This toy scorer stands in for encoder/predictor/joiner state.",
 "description_zh":"`score(t,prefix)` 根据帧 t 和已输出 token 元组 prefix 返回 V 个 logits。给定 input_length 帧、blank=0 与正整数 max_output_tokens，反复取 argmax（同分选最小 ID）：blank 推进 t，非 blank 追加 token 而不推进 t。到达末帧或 token 上限立即停止，返回 `(token 元组, 是否被上限截断)`；仅因上限停止时截断标记为 True。重复 token 保留。本题用简化评分函数代替编码器、预测网络和联合网络状态。",
 "function_name":"asr_rnnt_greedy", "hint":"Keep t and prefix separate; a nonblank output must not advance time.", "hint_zh":"分别维护 t 与 prefix；输出非 blank 时不能推进时间。",
 "tests":[
  {"name":"Blank advances and repeated tokens remain", "code":"""\nimport torch\ndef score(t,p):\n if t==0 and len(p)<2: return torch.tensor([0.,3.])\n return torch.tensor([4.,0.])\nassert {fn}(score,2,5)==((1,1),False)\n"""},
  {"name":"Output limit and tie handling", "code":"""\nimport torch\nassert {fn}(lambda t,p:torch.tensor([0.,1.]),3,2)==((1,1),True)\nassert {fn}(lambda t,p:torch.tensor([1.,1.]),2,2)==((),False)\n"""}],
 "solution":'''def asr_rnnt_greedy(score, input_length, max_output_tokens, blank=0):
    t = 0
    tokens = []
    while t < input_length:
        token = int(score(t, tuple(tokens)).argmax())
        if token == blank:
            t += 1
        else:
            tokens.append(token)
            if len(tokens) >= max_output_tokens:
                return tuple(tokens), True
    return tuple(tokens), False''',
 "demo":"""print(asr_rnnt_greedy(lambda t,p:torch.tensor([1.,0.]),3,5))"""
}
