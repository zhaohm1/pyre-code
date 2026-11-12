# CTC、AED 与 RNN-T 最小实验

运行环境：Python 3.10+、PyTorch。仓库已有 Python 环境时，在项目根目录运行：

```bash
node scripts/run-python.js experiments/ctc_rnnt_aed/run.py --exp all
node scripts/run-python.js experiments/ctc_rnnt_aed/run.py --exp Exp004
```

`--exp` 可选 `Exp001` 至 `Exp010`。每项先运行一个可核对的合成样例，再打印关键结果。代码集中在 [run.py](run.py)，便于沿着十项实验阅读。所有张量均为合成输入；数值只验证算法与实现，不代表真实语音识别质量、训练效率或流式延迟。

| 实验 | 平台题目 ID | 核心代码及原理 | 样例检查 |
| --- | --- | --- | --- |
| Exp001 CTC Collapse | `asr_ctc_collapse` | `ctc_collapse` 先对原始逐帧路径合并连续重复，再删除 blank；blank 隔开的同一 token 保留两次。 | `[1,1,0,1,2,2] → (1,1,2)` |
| Exp002 CTC Forward-Backward | `asr_ctc_forward` | `ctc_forward_backward` 插入 blank 构造扩展目标；前向表累加到各状态的路径，后向表累加剩余路径，合法跳跃不能跨过重复 token。 | 与逐条穷举路径概率一致，首帧前后向总和等于序列概率。 |
| Exp003 CTC Prefix Beam Search | `asr_ctc_prefix` | `ctc_prefix_beam` 为每个前缀分别记录以 blank 和非 blank 结尾的概率质量，扩展重复 token 时区分“留在原前缀”和“经 blank 新增 token”。 | 宽束下与路径穷举后按文本合并的概率一致。 |
| Exp004 Mini-AED | `asr_aed_minimal` | `MiniAED` 将历史 token 嵌入后做因果自注意力，再以声学状态为 key/value 做交叉注意力；Teacher Forcing 用真实前缀预测右移目标。 | 改动未来输入不改变较早位置输出，Loss 可反传。 |
| Exp005 AED Beam Search | `asr_aed_beam` | `aed_beam` 每轮扩展未结束候选、累计对数概率，EOS 候选停止扩展，同分按 token 序列排序。 | 宽度 2 能选出局部次优但全局更优的序列。 |
| Exp006 CTC + AED Rescoring | `asr_attention_rescore` | `rescore` 只给 CTC N-best 候选计算注意力分数，再按权重融合；它无法找回候选外的文本。 | 注意力分数改变 N-best 内的排序。 |
| Exp007 Mini-RNN-T | `asr_rnnt_minimal` | `MiniRNNT` 包含声学 Encoder、历史 token GRU Predictor 和 Joint；`rnnt_forward` 在 `(t,u)` 格点对 blank 与 token 两种转移求和。 | Joint 形状为 `[B,T,U+1,V]`，小格点 Loss 可反传。 |
| Exp008 RNN-T Greedy Search | `asr_rnnt_greedy` | `rnnt_greedy` 中 blank 推进 `t`，非 blank 更新 token 前缀而停留在同一声学帧。 | 同一帧可连续输出两个相同 token。 |
| Exp009 RNN-T Beam Search | `asr_rnnt_beam` | `rnnt_beam` 在每帧枚举有界 token 扩展，合并到达同一前缀的对数概率；blank 后推进到下一帧。 | 打印合并后的候选及其分数。 |
| Exp010 Streaming RNN-T | `asr_rnnt_streaming` | `CausalEncoder` 使用 GRU 状态作跨块声学缓存；`streaming_rnnt` 保留历史 token，逐块按 blank/token 转移解码。 | 一次输入与分三块输入得到相同 token、结束状态和缓存。 |

## 代码阅读顺序与边界

先看 Exp001–003，分清“路径概率”和“文本概率”；然后看 Exp004–006，分清 Teacher Forcing、推理搜索与候选重排；最后看 Exp007–010，跟踪 RNN-T 的时间轴 `t`、输出轴 `u` 和跨块状态。平台 Hint 会按模块作用、实现步骤和关键边界引导独立完成题目。

Exp004/007 只展示随机初始化模型的一次前向与反向传播，没有训练语音模型。Exp009 为小词表、有输出长度上限的教学搜索，不是生产级 RNN-T Beam Search。Exp010 只验证因果 GRU 的跨块状态等价性；未测量实际延迟，也未实现音频特征提取、复杂 Encoder Cache 或端侧部署。
