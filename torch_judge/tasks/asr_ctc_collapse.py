"""CTC path-collapse rule."""
TASK = {
 "title":"CTC Path Collapse", "title_zh":"CTC 路径折叠", "difficulty":"Easy",
 "description_en":"Given a sequence of integer frame IDs `path` and `blank=0`, collapse consecutive equal IDs first, then delete blanks. Return a tuple of token IDs. Thus [1,1,0,1] -> (1,1), while [1,1] -> (1,). Empty input returns (). This is the CTC path mapping only, not probability computation.",
 "description_zh":"给定逐帧整数 ID 序列 `path`，blank 默认为 0；先合并连续相同 ID，再删去 blank，返回 token ID 元组。因此 [1,1,0,1] -> (1,1)，[1,1] -> (1,)。空输入返回 ()。本题只实现 CTC 路径映射，不计算概率。",
 "function_name":"asr_ctc_collapse", "hint":"Compare each frame with the immediately preceding original frame before removing blanks.", "hint_zh":"删除 blank 前，先将每帧与原始路径的前一帧比较。",
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
