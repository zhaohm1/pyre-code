"""Character error rate for normalized ASR text."""
TASK = {
 "title":"Chinese ASR Character Error Rate", "title_zh":"中文 ASR 字错率", "difficulty":"Medium",
 "description_en":"For reference and hypothesis strings, normalize each with Unicode NFKC, casefold Latin letters, and remove all Unicode whitespace and punctuation (category P). Keep Chinese characters and other symbols. Compute Levenshtein character distance, then return `(distance, CER)` with CER=distance/max(1,len(normalized_reference)). Empty/empty is (0,0); a nonempty hypothesis against an empty reference has CER equal to its normalized length, which may exceed 1. This is a specified text metric, not a model-quality claim.",
 "description_zh":"给定参考文本和识别文本：先做 Unicode NFKC 归一化，对拉丁字母做 casefold，并删除所有 Unicode 空白和类别 P 的标点；保留汉字及其他符号。计算字符级 Levenshtein 距离，返回 `(编辑距离, CER)`，其中 CER=距离/max(1,归一化参考文本长度)。双空文本返回 (0,0)；参考为空而识别非空时 CER 等于归一化后识别字符数，可能大于 1。这只是明确定义的文本指标，不代表模型质量。",
 "function_name":"asr_cer", "hint":"Normalize both sides before dynamic programming; each insertion, deletion or substitution costs one.", "hint_zh":"先归一化两侧文本，再做动态规划；插入、删除、替换各计 1。",
 "tests":[
  {"name":"Chinese insertion, deletion and substitution", "code":"""\nassert {fn}('你好世界','你好世间')==(1,0.25)\nassert {fn}('你好世界','你好')==(2,0.5)\nassert {fn}('你好','你好啊')==(1,0.5)\n"""},
  {"name":"Normalization and empty references", "code":"""\nassert {fn}('Ａ，B！','a b')==(0,0.)\nassert {fn}('','')==(0,0.)\nassert {fn}('','你好')==(2,2.)\nassert {fn}('你好','')==(2,1.)\n"""}],
 "solution":'''def asr_cer(reference, hypothesis):
    import unicodedata
    def normalize(text):
        text = unicodedata.normalize('NFKC', text).casefold()
        return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))
    ref, hyp = normalize(reference), normalize(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, char in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char != other)))
        previous = current
    distance = previous[-1]
    return distance, distance / max(1, len(ref))''',
 "demo":"""print(asr_cer('你好世界','你好世间'))"""
}
