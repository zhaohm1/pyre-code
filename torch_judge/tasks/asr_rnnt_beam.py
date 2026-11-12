"""Time-synchronous RNN-T beam search with bounded same-frame emissions."""

TASK = {
    "title":"RNN-T Beam Search", "title_zh":"RNN-T 束搜索", "difficulty":"Hard",
    "description_en":"Implement asr_rnnt_beam(score, input_length, beam_size, max_symbols_per_frame, max_output_tokens, blank=0). score(t,prefix_tuple) returns V log probabilities. For each frame, start from the previous frame's prefix beam. At depths 0 through max_symbols_per_frame, each active prefix may emit blank to advance to the next frame; when below both symbol limits, it may emit each nonblank token and remain on the same frame. Merge probability mass of paths reaching the same prefix using logaddexp. Keep beam_size best active prefixes after each nonblank depth and beam_size best blank-completed prefixes after each frame, ranked by descending log probability then lexicographic prefix. Return final ranked `(prefix_tuple, log_probability)` pairs. Positive input_length, beam_size, per-frame and total token limits. This bounded search approximates the full RNN-T search; same-frame repeated tokens are retained.",
    "description_zh":"实现 asr_rnnt_beam(score, input_length, beam_size, max_symbols_per_frame, max_output_tokens, blank=0)。score(t,token 前缀元组) 返回 V 个对数概率。每帧从上一帧的前缀束开始，按深度 0 到 max_symbols_per_frame 扩展：活动前缀可发 blank 推进到下一帧；未达到两个 token 上限时，可发各非 blank token 并留在当前帧。用 logaddexp 合并到达同一前缀的路径概率。每次非 blank 扩展后保留 beam_size 个活动前缀，每帧 blank 汇合后再保留 beam_size 个前缀；均按概率降序、同分按前缀字典序。返回最终排序的 `(前缀元组, 对数概率)` 列表。帧数、束宽及两个 token 上限均为正。此有界搜索近似完整 RNN-T 搜索；同帧重复 token 保留。",
    "function_name":"asr_rnnt_beam",
    "hint":"A beam path has both acoustic time t and emitted-token prefix. At a fixed t, token edges create deeper same-frame states; blank edges collect paths for t+1. Sum equal-prefix path masses in log space before pruning. Force a blank edge at the per-frame symbol limit; never collapse repeated tokens as CTC does.",
    "hint_zh":"每条束路径同时有声学时间 t 和已输出 token 前缀。固定 t 时，非 blank 边产生更深的同帧状态，blank 边汇入 t+1。裁剪前先在对数域合并相同前缀的路径概率；达到每帧 token 上限时只能走 blank。重复 token 不像 CTC 那样折叠。",
    "tests":[
        {"name":"One-frame probabilities including repeated tokens", "code":"""
import torch,math
def score(t,p): return torch.tensor([math.log(.4),math.log(.6)])
got=dict({fn}(score,1,4,2,2))
assert set(got)=={(),(1,),(1,1)}
assert abs(math.exp(got[()])-.4)<1e-6
assert abs(math.exp(got[(1,)])-.24)<1e-6
assert abs(math.exp(got[(1,1)])-.144)<1e-6
"""},
        {"name":"Two-frame paths merge and tie ranking", "code":"""
import torch,math
def score(t,p): return torch.tensor([math.log(.5),math.log(.5)])
got={fn}(score,2,8,1,2)
scores=dict(got)
assert abs(math.exp(scores[(1,)])-.25)<1e-6
assert got==sorted(got,key=lambda x:(-x[1],x[0]))
assert all(len(p)<=2 for p,s in got)
"""},
    ],
    "solution":'''def asr_rnnt_beam(score, input_length, beam_size, max_symbols_per_frame, max_output_tokens, blank=0):
    import math
    def merge(table, prefix, value):
        old = table.get(prefix, float('-inf'))
        if old == float('-inf'):
            table[prefix] = value
        else:
            larger = max(old, value)
            table[prefix] = larger + math.log1p(math.exp(min(old, value) - larger))
    def top(table):
        return dict(sorted(table.items(), key=lambda item: (-item[1], item[0]))[:beam_size])
    beam = {(): 0.0}
    for t in range(input_length):
        active = beam
        advanced = {}
        for depth in range(max_symbols_per_frame + 1):
            next_active = {}
            for prefix, value in active.items():
                lp = score(t, prefix)
                merge(advanced, prefix, value + float(lp[blank]))
                if depth < max_symbols_per_frame and len(prefix) < max_output_tokens:
                    for token in range(lp.numel()):
                        if token != blank:
                            merge(next_active, prefix + (token,), value + float(lp[token]))
            active = top(next_active)
        beam = top(advanced)
    return list(beam.items())''',
    "demo":"""import torch
print(asr_rnnt_beam(lambda t,p:torch.tensor([-.2,-1.]),2,3,2,3))""",
}
