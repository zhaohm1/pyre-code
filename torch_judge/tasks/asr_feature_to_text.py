"""Synthetic feature-to-text ASR integration exercise."""
TASK = {
 "title":"ASR Feature-to-Text Walkthrough", "title_zh":"ASR 特征到文本综合练习", "difficulty":"Hard",
 "description_en":"Connect three toy decoding branches on a padded log-Mel batch `features` (B,T,F), `lengths` (B,), and `references` (B strings). `encoder(masked_features,lengths)` returns (B,T,D); `ctc_head(states)` returns (B,T,V) logits. For each item, CTC prefix-search the valid frames with blank 0, then rescore those candidates using `attention_score(valid_states,token_tuple)` and 0<=ctc_weight<=1. Separately, RNN-T greedy-decode `rnnt_score(valid_states,t,prefix)` with a positive `max_output_tokens`. `vocab` maps ID to string, with vocab[0]=''. Return one dict per item with `ctc`, `rnnt`, `rescored` strings, `cer` of rescored vs reference, and `rnnt_truncated` bool. Use positive `beam_size`; ties choose lexicographically smaller token tuples. Length zero yields empty outputs. CER normalizes NFKC, casefold, and removes Unicode whitespace/punctuation. This synthetic chain does not train or represent a single full ASR model.",
 "description_zh":"连接合成 ASR 的三条解码分支。输入为填充的 log-Mel 特征 `features` (B,T,F)、长度 `lengths` (B,) 和参考文本 `references`（B 个字符串）。`encoder(masked_features,lengths)` 返回 (B,T,D)，`ctc_head(states)` 返回 (B,T,V) logits。每条语音仅在有效帧上做 blank=0 的 CTC 前缀搜索，再用 `attention_score(valid_states,token 元组)` 按 0<=ctc_weight<=1 重排候选；另用 `rnnt_score(valid_states,t,prefix)` 做 RNN-T 贪心解码，受正整数 `max_output_tokens` 限制。`vocab` 将 ID 映射为字符串，vocab[0]=''。每条返回含 `ctc`、`rnnt`、`rescored` 文本、重打分文本相对参考的 `cer` 和布尔值 `rnnt_truncated` 的字典。beam_size 为正数；同分选字典序较小的 token 元组。有效长度为零时返回空文本。CER 做 NFKC、casefold，删除 Unicode 空白与标点。此合成链路不训练模型，也不表示单个完整 ASR 模型。",
 "function_name":"asr_feature_to_text", "hint":"Mask before the encoder; slice valid frames for every decoder; merge CTC paths by prefix, rescore candidates, then map IDs to text and compute CER.", "hint_zh":"编码器前先遮蔽填充帧；各解码器只看有效帧；CTC 按前缀合并路径，候选重打分后映射为文本并计算 CER。",
 "tests":[
  {"name":"Three branches and attention changes CTC winner", "code":"""\nimport torch\nx=torch.tensor([[[0.,3.,1.],[3.,0.,0.]],[[0.,0.,4.],[90.,90.,90.]]])\ndef enc(z,lengths): return z\ndef head(z): return z\ndef rnnt(states,t,prefix): return torch.tensor([5.,0.,0.]) if prefix else torch.tensor([0.,4.,0.])\ndef att(states,prefix): return 20. if prefix==(2,) else 0.\nr={fn}(x,torch.tensor([2,1]),enc,head,rnnt,att,['','你','好'],['好','好'],beam_size=3,max_output_tokens=3,ctc_weight=0.1)\nassert len(r)==2 and r[0]['ctc']=='你' and r[0]['rnnt']=='你' and r[0]['rescored']=='好' and r[0]['cer']==0.\nassert r[1]['ctc']=='好' and r[1]['rescored']=='好' and r[1]['cer']==0.\nassert not r[0]['rnnt_truncated'] and not r[1]['rnnt_truncated']\n"""},
  {"name":"Zero length, padding invariance and output limit", "code":"""\nimport torch\nx=torch.tensor([[[1.,2.],[80.,90.]],[[4.,0.],[5.,0.]]]); lengths=torch.tensor([0,1])\ndef enc(z,l): return z\ndef head(z): return z\ndef rnnt(states,t,prefix): return torch.tensor([0.,3.])\ndef att(states,prefix): return 0.\na={fn}(x,lengths,enc,head,rnnt,att,['','中'],['','中'],2,2,0.5)\nx[0,1]=torch.tensor([-999.,999.]); x[1,1]=torch.tensor([999.,999.])\nb={fn}(x,lengths,enc,head,rnnt,att,['','中'],['','中'],2,2,0.5)\nassert a==b and a[0]['ctc']==a[0]['rnnt']==a[0]['rescored']=='' and a[0]['cer']==0.\nassert a[1]['rnnt']=='中中' and a[1]['rnnt_truncated']\n"""},
  {"name":"CER normalization and deterministic candidate ties", "code":"""\nimport torch\nx=torch.zeros(1,1,3); x[0,0,1:]=2.\nr={fn}(x,torch.tensor([1]),lambda z,l:z,lambda z:z,lambda s,t,p:torch.tensor([3.,0.,0.]),lambda s,p:0.,['','Ａ','B'],['a'],2,2,0.)\nassert r[0]['ctc']=='Ａ' and r[0]['rescored']=='Ａ' and r[0]['cer']==0.\n"""},
  {"name":"Padding is cleared before a context-mixing encoder", "code":"""\nimport torch\nx=torch.tensor([[[0.,4.],[100.,0.]]]); lengths=torch.tensor([1])\ndef enc(z,l): return z+z.sum(1,keepdim=True)\ndef head(z): return torch.cat([torch.zeros_like(z[...,:1]),z],-1)\ndef rnnt(s,t,p): return torch.tensor([3.,0.,0.])\na={fn}(x,lengths,enc,head,rnnt,lambda s,p:0.,['','你','好'],['好'],2,2,1.)\nx[0,1]=torch.tensor([-999.,999.])\nb={fn}(x,lengths,enc,head,rnnt,lambda s,p:0.,['','你','好'],['好'],2,2,1.)\nassert a==b and a[0]['ctc']=='好'\n"""}],
 "solution":'''def asr_feature_to_text(features, lengths, encoder, ctc_head, rnnt_score, attention_score, vocab, references, beam_size, max_output_tokens, ctc_weight):
    import torch, math, unicodedata
    valid = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
    states = encoder(features * valid.unsqueeze(-1).to(features.dtype), lengths)
    logits = ctc_head(states)
    def text(tokens):
        return ''.join(vocab[token] for token in tokens)
    def cer(reference, hypothesis):
        def norm(value):
            value = unicodedata.normalize('NFKC', value).casefold()
            return ''.join(c for c in value if not c.isspace() and not unicodedata.category(c).startswith('P'))
        ref, hyp = norm(reference), norm(hypothesis)
        previous = list(range(len(hyp) + 1))
        for i, char in enumerate(ref, 1):
            current = [i]
            for j, other in enumerate(hyp, 1):
                current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (char != other)))
            previous = current
        return previous[-1] / max(1, len(ref))
    results = []
    for b, length in enumerate(lengths):
        n = int(length)
        hidden = states[b, :n]
        if n == 0:
            candidates = [((), 0.0)]
        else:
            beam = {(): (1.0, 0.0)}
            frame_probs = logits[b, :n].log_softmax(-1).exp().tolist()
            for probs in frame_probs:
                next_beam = {}
                def add(prefix, blank_add=0.0, nonblank_add=0.0):
                    pb, pn = next_beam.get(prefix, (0.0, 0.0))
                    next_beam[prefix] = (pb + blank_add, pn + nonblank_add)
                for prefix, (pb, pn) in beam.items():
                    add(prefix, blank_add=(pb + pn) * probs[0])
                    for token in range(1, len(vocab)):
                        if prefix and token == prefix[-1]:
                            add(prefix, nonblank_add=pn * probs[token])
                            add(prefix + (token,), nonblank_add=pb * probs[token])
                        else:
                            add(prefix + (token,), nonblank_add=(pb + pn) * probs[token])
                ranked = sorted((p for p in next_beam if sum(next_beam[p]) > 0), key=lambda p: (-(next_beam[p][0] + next_beam[p][1]), p))
                beam = {p: next_beam[p] for p in ranked[:beam_size]}
            candidates = [(p, math.log(pb + pn)) for p, (pb, pn) in beam.items()]
        rescored = sorted(candidates, key=lambda item: (-(ctc_weight * item[1] + (1 - ctc_weight) * float(attention_score(hidden, item[0]))), item[0]))
        t, tokens, truncated = 0, [], False
        while t < n:
            token = int(rnnt_score(hidden, t, tuple(tokens)).argmax())
            if token == 0:
                t += 1
            else:
                tokens.append(token)
                if len(tokens) >= max_output_tokens:
                    truncated = True
                    break
        best_text = text(rescored[0][0])
        results.append({'ctc': text(candidates[0][0]), 'rnnt': text(tokens), 'rescored': best_text,
                        'cer': cer(references[b], best_text), 'rnnt_truncated': truncated})
    return results''',
 "demo":"""x=torch.randn(1,3,3)\nprint(asr_feature_to_text(x,torch.tensor([3]),lambda z,l:z,lambda z:z,lambda s,t,p:torch.tensor([1.,0.,0.]),lambda s,p:0.,['','你','好'],['你'],3,4,0.5))"""
}
