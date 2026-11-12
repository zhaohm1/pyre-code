"""CTC path-collapse rule."""
TASK = {
 "title":"CTC Path Collapse", "title_zh":"CTC 路径折叠", "difficulty":"Easy",
 "description_en":"Given a sequence of integer frame IDs `path` and `blank=0`, collapse consecutive equal IDs first, then delete blanks. Return a tuple of token IDs. Thus [1,1,0,1] -> (1,1), while [1,1] -> (1,). Empty input returns (). This is the CTC path mapping only, not probability computation.",
 "description_zh":"给定逐帧整数 ID 序列 `path`，blank 默认为 0；先合并连续相同 ID，再删去 blank，返回 token ID 元组。因此 [1,1,0,1] -> (1,1)，[1,1] -> (1,)。空输入返回 ()。本题只实现 CTC 路径映射，不计算概率。",
 "function_name":"asr_ctc_collapse", "hint":"CTC maps a frame-level alignment to a token sequence; blank separates repeated tokens. Walk the original path in order and track the previous frame ID, including blanks. Emit a nonblank ID only when it differs from that previous ID; update the previous ID on every frame. Why must [1,0,1] keep both 1s while [1,1] keeps one? Empty input yields an empty tuple.", "hint_zh":"CTC 将逐帧对齐路径映射为 token 序列；blank 可以隔开重复 token。按原始路径逐帧遍历，记录上一帧 ID（包括 blank）。仅当当前 ID 非 blank 且不同于上一帧时输出，并在每一帧更新上一帧 ID。想一想为何 [1,0,1] 保留两个 1，而 [1,1] 只保留一个；空输入返回空元组。",
 "tests":[
  {"name":"Blank-separated repeats", "code":"""\nassert {fn}([1,1,0,1])==(1,1)\nassert {fn}([1,1,1])==(1,)\nassert {fn}([0,0,2,2,0,3,3])==(2,3)\n"""},
  {"name":"Empty and custom blank", "code":"""\nassert {fn}([])==() and {fn}([0,0])==()\nassert {fn}([2,2,1,2],blank=2)==(1,)\n"""}],
 "solution":'''def asr_ctc_collapse(path, blank=0):
    out = []
    previous = None
    for token in path:
        if token != previous and token != blank:
            out.append(token)
        previous = token
    return tuple(out)''',
 "demo":"""print(asr_ctc_collapse([1,1,0,1,2,2]))"""
}
