"""Autoregressive beam search for a toy AED."""

TASK = {
    "title":"AED Beam Search", "title_zh":"AED 束搜索", "difficulty":"Medium",
    "description_en":"Implement asr_aed_beam(score, bos, eos, max_output_tokens, beam_size). score(prefix_tuple) returns V log probabilities for the next token, where prefix includes BOS. At each step expand every unfinished hypothesis by every vocabulary token, add log probabilities, keep the best beam_size hypotheses by descending score then lexicographic token tuple. EOS completes a hypothesis and is never expanded again. Stop when all retained hypotheses complete or max_output_tokens tokens after BOS have been generated. Return (best_tokens_tuple, best_log_score, completed_bool), excluding BOS and EOS from best_tokens_tuple. Use the same ranking for final selection; positive max_output_tokens and beam_size, distinct BOS/EOS.",
    "description_zh":"实现 asr_aed_beam(score, bos, eos, max_output_tokens, beam_size)。score(prefix_tuple) 根据含 BOS 的历史 token 返回下一 token 的 V 个对数概率。每步扩展所有未结束候选，累加对数概率，再按分数降序、同分按 token 元组字典序保留 beam_size 个。EOS 使候选结束，之后不再扩展。全部保留候选结束或 BOS 后已生成 max_output_tokens 个 token 时停止。返回 (最佳 token 元组, 最佳对数分数, 是否到达 EOS)，输出 token 元组不含 BOS/EOS。最终仍按相同顺序选最佳候选；两个上限均为正，BOS 与 EOS 不同。",
    "function_name":"asr_aed_beam",
    "hint":"AED feeds each predicted token back to the decoder. A beam stores (prefix, cumulative_log_score, finished), starting with BOS and score zero. Only unfinished beams call score; EOS marks a completed path. Rank complete and incomplete candidates together after each expansion, and strip BOS/EOS only when returning the winner.",
    "hint_zh":"AED 将预测 token 反馈给解码器。每个束保存 (历史元组, 累计对数分数, 是否结束)，从 BOS 和分数 0 开始。仅未结束的候选调用 score；EOS 标记完成路径。每步把已结束与新扩展候选一起排序，最终返回时才去掉 BOS/EOS。",
    "tests":[
        {"name":"Beam recovers a better complete path", "code":"""
import torch
def score(p):
 if len(p)==1: return torch.tensor([-100.,-1.,-.5,-100.])
 if p[-1]==1: return torch.tensor([-100.,-100.,-100.,0.])
 return torch.tensor([-100.,-100.,-100.,-10.])
assert {fn}(score,0,3,2,2)==((1,),-1.,True)
"""},
        {"name":"EOS is not expanded; tie and token limit", "code":"""
import torch
calls=[]
def score(p):
 calls.append(p)
 return torch.tensor([-100.,0.,0.,0.])
tokens,s,done={fn}(score,0,3,2,2)
assert tokens==(1,1) and s==0. and not done
assert all(p[-1]!=3 for p in calls)
assert {fn}(lambda p:torch.tensor([-100.,-100.,-100.,0.]),0,3,3,2)==((),0.,True)
"""},
    ],
    "solution":'''def asr_aed_beam(score, bos, eos, max_output_tokens, beam_size):
    beams = [((bos,), 0.0, False)]
    for _ in range(max_output_tokens):
        candidates = []
        for prefix, value, finished in beams:
            if finished:
                candidates.append((prefix, value, True))
                continue
            log_probs = score(prefix)
            for token in range(log_probs.numel()):
                new_prefix = prefix + (token,)
                candidates.append((new_prefix, value + float(log_probs[token]), token == eos))
        beams = sorted(candidates, key=lambda item: (-item[1], item[0]))[:beam_size]
        if all(item[2] for item in beams):
            break
    prefix, value, finished = beams[0]
    return prefix[1:-1] if finished else prefix[1:], value, finished''',
    "demo":"""import torch
print(asr_aed_beam(lambda p:torch.tensor([-5.,-2.,0.]),0,2,3,2))""",
}
