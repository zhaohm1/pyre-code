"""Chunked causal encoder and RNN-T greedy state."""

TASK = {
    "title":"Streaming RNN-T State", "title_zh":"流式 RNN-T 状态缓存", "difficulty":"Hard",
    "description_en":"Implement asr_rnnt_streaming(chunks, encoder_step, score, max_symbols_per_frame, max_output_tokens, blank=0). chunks is a sequence of (C,F) tensors, possibly including empty chunks. encoder_step(frame, cache) returns (hidden, new_cache), starting with cache=None; it must be called once per frame in order. score(hidden, prefix_tuple) returns V logits. Greedily choose argmax (lowest ID on ties): blank advances to the next encoded frame; nonblank appends a token and stays on this frame. Keep the encoder cache and token prefix across chunk boundaries. Force frame advance after max_symbols_per_frame nonblank emissions on that frame; stop immediately at max_output_tokens total emissions. Return (tokens_tuple, final_cache, truncated_bool), where truncated is true only if total output limit stopped decoding. Both limits are positive. This tests state handling with a caller-supplied causal encoder, not real-time latency.",
    "description_zh":"实现 asr_rnnt_streaming(chunks, encoder_step, score, max_symbols_per_frame, max_output_tokens, blank=0)。chunks 是若干 (C,F) 张量，可含空块。encoder_step(frame, cache) 返回 (hidden, new_cache)，初始 cache=None；应按顺序对每帧调用一次。score(hidden, token 前缀元组) 返回 V 个 logits。贪心取 argmax（同分选最小 ID）：blank 推进到下一编码帧，非 blank 追加 token 并留在本帧。跨块保留 encoder cache 与 token 前缀。每帧输出 max_symbols_per_frame 个非 blank 后强制推进；总输出达到 max_output_tokens 时立即停止。返回 (token 元组, 最终 cache, 是否截断)，仅因总输出上限而停止时截断为真。两个上限均为正。本题用调用方给定的因果 encoder 检验状态维护，不验证真实流式延迟。",
    "function_name":"asr_rnnt_streaming",
    "hint":"The encoder cache carries acoustic history; the emitted prefix is the prediction-network state in this toy interface. Process every chunk's frames in order, passing the updated cache to the next frame. For each hidden frame, query the joint scorer repeatedly until blank or the per-frame symbol limit. A token does not advance the frame or reset cache. The total output limit ends the whole stream, including remaining chunks.",
    "hint_zh":"Encoder cache 保存声学历史；本题用已输出前缀代替 Prediction Network 状态。按块、按帧顺序处理，每帧把更新后的 cache 传给下一帧。对同一 hidden 帧反复查询 Joint 评分，直到 blank 或该帧 token 上限；输出 token 不推进帧，也不重置 cache。总输出上限会结束整条流，剩余块不再处理。",
    "tests": [
        {"name":"Partition invariance and cross-chunk cache", "code":"""
import torch
x=torch.tensor([[1.],[2.],[3.],[4.]])
def enc(frame,cache):
 state=frame if cache is None else cache+frame
 return state,state
def score(hidden,prefix):
 if int(hidden.item())%2 and (not prefix or prefix[-1]!=1): return torch.tensor([0.,2.])
 return torch.tensor([2.,0.])
a={fn}([x],enc,score,2,10); b={fn}([x[:1],x[1:3],x[3:]],enc,score,2,10)
assert a[0]==b[0] and torch.equal(a[1],b[1]) and not a[2] and not b[2]
assert torch.equal(a[1],torch.tensor([10.]))
"""},
        {"name":"Empty chunks, repeated tokens and global limit", "code":"""
import torch
calls=[]
def enc(frame,cache):
 calls.append(float(frame[0])); return frame,frame
def score(h,p): return torch.tensor([0.,1.])
x=torch.tensor([[1.],[2.],[3.]])
tokens,cache,truncated={fn}([x[:0],x[:1],x[1:]],enc,score,3,2)
assert tokens==(1,1) and truncated and calls==[1.]
tokens,cache,truncated={fn}([x[:0]],enc,score,3,2)
assert tokens==() and cache is None and not truncated
"""},
        {"name":"Blank advances frames and ties prefer blank", "code":"""
import torch
x=torch.ones(3,1)
def enc(frame,cache): return frame,(0 if cache is None else cache)+1
assert {fn}([x[:1],x[1:]],enc,lambda h,p:torch.tensor([1.,1.]),2,5)==((),3,False)
"""},
    ],
    "solution":'''def asr_rnnt_streaming(chunks, encoder_step, score, max_symbols_per_frame, max_output_tokens, blank=0):
    cache = None
    tokens = []
    for chunk in chunks:
        for frame in chunk:
            hidden, cache = encoder_step(frame, cache)
            for _ in range(max_symbols_per_frame):
                token = int(score(hidden, tuple(tokens)).argmax())
                if token == blank:
                    break
                tokens.append(token)
                if len(tokens) >= max_output_tokens:
                    return tuple(tokens), cache, True
    return tuple(tokens), cache, False''',
    "demo":"""import torch
def enc(frame,cache):
 cache=frame if cache is None else cache+frame
 return cache,cache
print(asr_rnnt_streaming([torch.ones(2,1)],enc,lambda h,p:torch.tensor([2.,0.]),2,5))""",
}
