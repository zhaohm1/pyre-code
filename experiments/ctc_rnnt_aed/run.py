"""Ten small, deterministic CTC/AED/RNN-T experiments. Run with --exp Exp001."""
from __future__ import annotations

import argparse
import itertools
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_collapse(path, blank=0):
    result, previous = [], None
    for token in path:
        if token != previous and token != blank:
            result.append(token)
        previous = token
    return tuple(result)


def ctc_forward_backward(log_probs, target, blank=0):
    """Return log P(target), frame/state forward table and backward table."""
    t_max = log_probs.shape[0]
    extended = [blank]
    for token in target:
        extended += [token, blank]
    s_max = len(extended)
    neg = log_probs.new_tensor(float('-inf'))
    forward = [[neg for _ in extended] for _ in range(t_max)]
    forward[0][0] = log_probs[0, blank]
    if s_max > 1:
        forward[0][1] = log_probs[0, extended[1]]
    for t in range(1, t_max):
        for s, token in enumerate(extended):
            sources = [forward[t - 1][s]]
            if s:
                sources.append(forward[t - 1][s - 1])
            if s > 1 and token != blank and token != extended[s - 2]:
                sources.append(forward[t - 1][s - 2])
            forward[t][s] = torch.logsumexp(torch.stack(sources), 0) + log_probs[t, token]
    backward = [[neg for _ in extended] for _ in range(t_max)]
    backward[-1][-1] = log_probs.new_tensor(0.)
    if s_max > 1:
        backward[-1][-2] = log_probs.new_tensor(0.)
    for t in range(t_max - 2, -1, -1):
        for s in range(s_max):
            transitions = [log_probs[t + 1, extended[s]] + backward[t + 1][s]]
            if s + 1 < s_max:
                transitions.append(log_probs[t + 1, extended[s + 1]] + backward[t + 1][s + 1])
            if s + 2 < s_max and extended[s + 2] != blank and extended[s] != extended[s + 2]:
                transitions.append(log_probs[t + 1, extended[s + 2]] + backward[t + 1][s + 2])
            backward[t][s] = torch.logsumexp(torch.stack(transitions), 0)
    total = torch.logsumexp(torch.stack(forward[-1][-2:] if s_max > 1 else forward[-1]), 0)
    return total, torch.stack([torch.stack(row) for row in forward]), torch.stack([torch.stack(row) for row in backward])


def ctc_prefix_beam(log_probs, beam_size, blank=0):
    """Prefix masses are stored as (blank ending, nonblank ending)."""
    beams = {(): (1., 0.)}
    for frame in log_probs.exp().tolist():
        next_beams = defaultdict(lambda: [0., 0.])
        for prefix, (pb, pn) in beams.items():
            next_beams[prefix][0] += (pb + pn) * frame[blank]
            for token, probability in enumerate(frame):
                if token == blank:
                    continue
                if prefix and token == prefix[-1]:
                    next_beams[prefix][1] += pn * probability
                    next_beams[prefix + (token,)][1] += pb * probability
                else:
                    next_beams[prefix + (token,)][1] += (pb + pn) * probability
        order = sorted((p for p in next_beams if sum(next_beams[p]) > 0), key=lambda p: (-sum(next_beams[p]), p))[:beam_size]
        beams = {p: tuple(next_beams[p]) for p in order}
    return [(p, math.log(sum(beams[p]))) for p in sorted(beams, key=lambda p: (-sum(beams[p]), p))]


class MiniAED(nn.Module):
    def __init__(self, vocab=5, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.self_attention = nn.MultiheadAttention(dim, 2, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, 2, batch_first=True)
        self.output = nn.Linear(dim, vocab)

    def forward(self, acoustic, decoder_input):
        x = self.embed(decoder_input)
        length = x.shape[1]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        x, _ = self.self_attention(x, x, x, attn_mask=mask)
        x, weights = self.cross_attention(x, acoustic, acoustic)
        return self.output(x), weights


def aed_beam(score, bos, eos, beam_size, max_tokens):
    beams = [((), 0., False)]
    for _ in range(max_tokens):
        candidates = []
        for sequence, log_score, ended in beams:
            if ended:
                candidates.append((sequence, log_score, True))
                continue
            for token, value in enumerate(score((bos,) + sequence).tolist()):
                candidates.append((sequence + (token,), log_score + value, token == eos))
        beams = sorted(candidates, key=lambda x: (-x[1], x[0]))[:beam_size]
        if all(ended for _, _, ended in beams):
            break
    return beams


def rescore(candidates, attention_score, ctc_weight):
    return sorted(((tokens, ctc_weight * ctc + (1 - ctc_weight) * attention_score(tokens))
                   for tokens, ctc in candidates), key=lambda x: (-x[1], x[0]))


class MiniRNNT(nn.Module):
    def __init__(self, acoustic_dim=3, vocab=4, dim=8):
        super().__init__()
        self.encoder = nn.Linear(acoustic_dim, dim)
        self.embedding = nn.Embedding(vocab, dim)
        self.predictor = nn.GRU(dim, dim, batch_first=True)
        self.joint = nn.Linear(dim, vocab)

    def forward(self, acoustic, history):
        encoded = self.encoder(acoustic)
        predicted, _ = self.predictor(self.embedding(history))
        return self.joint(torch.tanh(encoded[:, :, None, :] + predicted[:, None, :, :]))


def rnnt_forward(log_probs, target, blank=0):
    t_max, u_states, _ = log_probs.shape
    assert u_states == len(target) + 1
    lattice = [[log_probs.new_tensor(float('-inf')) for _ in range(u_states)] for _ in range(t_max + 1)]
    lattice[0][0] = log_probs.new_tensor(0.)
    for t in range(t_max):
        for u in range(u_states):
            lattice[t + 1][u] = torch.logaddexp(lattice[t + 1][u], lattice[t][u] + log_probs[t, u, blank])
            if u < len(target):
                lattice[t][u + 1] = torch.logaddexp(lattice[t][u + 1], lattice[t][u] + log_probs[t, u, target[u]])
    return lattice[-1][-1]


def rnnt_greedy(score, frames, max_tokens, blank=0):
    tokens, t = (), 0
    while t < frames:
        token = int(torch.argmax(score(t, tokens)))
        if token == blank:
            t += 1
        else:
            tokens += (token,)
            if len(tokens) == max_tokens:
                return tokens, True
    return tokens, False


def rnnt_beam(score, frames, beam_size, max_tokens, blank=0):
    """Bounded small-example search, merging identical token prefixes at each frame."""
    beams = {(): 0.}
    for t in range(frames):
        pending = dict(beams)
        completed = {}
        for _ in range(max_tokens + 1):
            if not pending:
                break
            extensions = {}
            for prefix, mass in pending.items():
                probabilities = score(t, prefix).tolist()
                ended = mass + probabilities[blank]
                completed[prefix] = torch.logaddexp(torch.tensor(completed.get(prefix, -math.inf)), torch.tensor(ended)).item()
                if len(prefix) < max_tokens:
                    for token, value in enumerate(probabilities):
                        if token != blank:
                            child = prefix + (token,)
                            extensions[child] = torch.logaddexp(torch.tensor(extensions.get(child, -math.inf)), torch.tensor(mass + value)).item()
            pending = extensions
        order = sorted(completed, key=lambda p: (-completed[p], p))[:beam_size]
        beams = {p: completed[p] for p in order}
    return sorted(beams.items(), key=lambda x: (-x[1], x[0]))


class CausalEncoder(nn.Module):
    def __init__(self, feature_dim=3, hidden_dim=4):
        super().__init__()
        self.rnn = nn.GRU(feature_dim, hidden_dim, batch_first=True)

    def forward_chunk(self, chunk, cache=None):
        return self.rnn(chunk, cache)


def streaming_rnnt(chunks, encoder, score, max_tokens=8, max_symbols_per_frame=4, blank=0):
    cache, output = None, ()
    for chunk in chunks:
        encoded, cache = encoder.forward_chunk(chunk, cache)
        for t in range(encoded.shape[1]):
            emitted_on_frame = 0
            while emitted_on_frame < max_symbols_per_frame:
                token = int(torch.argmax(score(encoded[:, t], output)))
                if token == blank:
                    break
                output += (token,)
                emitted_on_frame += 1
                if len(output) >= max_tokens:
                    return output, cache, True
    return output, cache, False


def run(exp):
    torch.manual_seed(7)
    if exp == 'Exp001':
        assert ctc_collapse([1, 1, 0, 1, 2, 2]) == (1, 1, 2)
        print('blank-separated repeats:', ctc_collapse([1, 1, 0, 1, 2, 2]))
    elif exp == 'Exp002':
        p = torch.tensor([[.2, .8], [.6, .4], [.3, .7]])
        total, alpha, beta = ctc_forward_backward(p.log(), [1])
        brute = sum(math.prod(p[t, token].item() for t, token in enumerate(path))
                    for path in itertools.product(range(2), repeat=3) if ctc_collapse(path) == (1,))
        assert abs(total.exp().item() - brute) < 1e-6
        assert torch.allclose(torch.logsumexp(alpha[0, :2] + beta[0, :2], 0), total)
        print('P(1)=', total.exp().item(), 'forward states=', tuple(alpha.shape))
    elif exp == 'Exp003':
        p = torch.tensor([[.2, .5, .3], [.4, .4, .2], [.3, .2, .5]])
        beam = ctc_prefix_beam(p.log(), 27)
        brute = defaultdict(float)
        for path in itertools.product(range(3), repeat=3):
            brute[ctc_collapse(path)] += math.prod(p[t, token].item() for t, token in enumerate(path))
        assert all(abs(math.exp(s) - brute[prefix]) < 1e-6 for prefix, s in beam)
        print('best prefixes:', beam[:3])
    elif exp == 'Exp004':
        model = MiniAED()
        acoustic = torch.randn(1, 4, 8)
        history = torch.tensor([[1, 2, 3]])
        logits, attention = model(acoustic, history)
        loss = F.cross_entropy(logits.reshape(-1, 5), torch.tensor([[2, 3, 4]]).reshape(-1))
        changed = history.clone(); changed[0, 2] = 4
        other, _ = model(acoustic, changed)
        assert logits.shape == (1, 3, 5) and attention.shape == (1, 3, 4)
        assert torch.allclose(logits[:, :2], other[:, :2], atol=1e-6)
        loss.backward()
        print('teacher-forcing loss:', loss.item(), 'cross-attention:', tuple(attention.shape))
    elif exp == 'Exp005':
        def score(seq):
            values = [-9., -1., -.5, -9.] if len(seq) == 1 else ([-9., -9., -9., 0.] if seq[-1] == 1 else [-9., -9., -9., -10.])
            return torch.tensor(values)
        result = aed_beam(score, bos=0, eos=3, beam_size=2, max_tokens=2)
        assert result[0][0] == (1, 3)
        print('AED beam:', result)
    elif exp == 'Exp006':
        candidates = [((1,), -1.), ((2,), -1.2)]
        ranked = rescore(candidates, lambda p: 1. if p == (2,) else -1., .3)
        assert ranked[0][0] == (2,) and all(p in dict(candidates) for p, _ in ranked)
        print('rescored N-best:', ranked)
    elif exp == 'Exp007':
        model = MiniRNNT()
        logits = model(torch.randn(1, 2, 3), torch.tensor([[0, 1]]))
        assert logits.shape == (1, 2, 2, 4)
        loss = -rnnt_forward(logits[0].log_softmax(-1), [1])
        assert torch.isfinite(loss)
        loss.backward()
        print('joint shape:', tuple(logits.shape), 'RNNT loss:', loss.item())
    elif exp == 'Exp008':
        score = lambda t, p: torch.tensor([0., 2.]) if t == 0 and len(p) < 2 else torch.tensor([3., 0.])
        output = rnnt_greedy(score, frames=2, max_tokens=5)
        assert output == ((1, 1), False)
        print('RNN-T greedy:', output)
    elif exp == 'Exp009':
        def score(t, prefix):
            if prefix == ():
                return torch.tensor([math.log(.4), math.log(.6)])
            return torch.tensor([math.log(.8), math.log(.2)])
        result = rnnt_beam(score, frames=2, beam_size=8, max_tokens=2)
        assert result[0][0] == (1,)
        print('RNN-T beam:', result[:3])
    elif exp == 'Exp010':
        encoder = CausalEncoder()
        x = torch.randn(1, 5, 3)
        def score(frame, prefix):
            return torch.tensor([2., 0.]) if len(prefix) >= 1 else torch.tensor([0., 2.])
        full = streaming_rnnt([x], encoder, score)
        chunks = streaming_rnnt([x[:, :2], x[:, 2:4], x[:, 4:]], encoder, score)
        assert full[0] == chunks[0] and full[2] == chunks[2]
        assert torch.allclose(full[1], chunks[1], atol=1e-6)
        print('streaming output:', chunks[0], 'cache shape:', tuple(chunks[1].shape))
    else:
        raise ValueError(exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', choices=[f'Exp{i:03d}' for i in range(1, 11)] + ['all'], default='all')
    args = parser.parse_args()
    for name in ([f'Exp{i:03d}' for i in range(1, 11)] if args.exp == 'all' else [args.exp]):
        print(name, end=': ')
        run(name)
