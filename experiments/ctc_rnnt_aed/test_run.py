"""Independent small-case checks for the lab algorithms."""
import itertools
import math
import unittest

import torch

from run import (ctc_collapse, ctc_forward_backward, ctc_prefix_beam,
                 rnnt_forward, rnnt_beam, aed_beam, rescore)


class TestASRLabs(unittest.TestCase):
    def test_ctc_forward_backward_and_prefix_against_enumeration(self):
        p = torch.tensor([[.25, .5, .25], [.4, .3, .3], [.2, .3, .5]])
        masses = {}
        for path in itertools.product(range(3), repeat=3):
            prefix = ctc_collapse(path)
            masses[prefix] = masses.get(prefix, 0.) + math.prod(p[t, token].item() for t, token in enumerate(path))
        for target in [(), (1,), (1, 1), (1, 2)]:
            total, alpha, beta = ctc_forward_backward(p.log(), target)
            self.assertAlmostEqual(total.exp().item(), masses.get(target, 0.), places=6)
            initial = torch.logsumexp(alpha[0, :min(2, alpha.shape[1])] + beta[0, :min(2, beta.shape[1])], 0)
            self.assertTrue(torch.allclose(total, initial, atol=1e-6))
        got = dict(ctc_prefix_beam(p.log(), 27))
        self.assertEqual(set(got), set(masses))
        for prefix, mass in masses.items():
            self.assertAlmostEqual(math.exp(got[prefix]), mass, places=6)

    def test_rnnt_forward_against_path_enumeration(self):
        p = torch.tensor([[[.7, .3], [.4, .6]], [[.2, .8], [.5, .5]]])
        # A single label may be emitted at either frame, then both frames emit blank.
        expected = (p[0, 0, 1] * p[0, 1, 0] * p[1, 1, 0]
                    + p[0, 0, 0] * p[1, 0, 1] * p[1, 1, 0])
        self.assertTrue(torch.allclose(rnnt_forward(p.log(), [1]).exp(), expected))

    def test_rnnt_beam_merges_paths(self):
        probabilities = {(): [.5, .5], (1,): [.8, .2]}
        score = lambda t, prefix: torch.tensor(probabilities.get(prefix, [.9, .1])).log()
        got = dict(rnnt_beam(score, frames=2, beam_size=20, max_tokens=2))
        # (1,) has paths: token at t0; or blank at t0 then token at t1.
        expected = .5 * .8 * .8 + .5 * .5 * .8
        self.assertAlmostEqual(math.exp(got[(1,)]), expected, places=6)
        self.assertEqual(rnnt_beam(score, frames=2, beam_size=2, max_tokens=2)[0][0], (1,))

    def test_aed_eos_and_rescore_candidate_boundary(self):
        def score(prefix):
            return torch.tensor([-9., -1., -.5, -9.]) if len(prefix) == 1 else torch.tensor([-9., -9., -9., 0.])
        result = aed_beam(score, bos=0, eos=3, beam_size=2, max_tokens=4)
        self.assertTrue(result[0][2])
        self.assertEqual(result[0][0], (2, 3))
        candidates = [((1,), -1.), ((2,), -1.2)]
        ranked = rescore(candidates, lambda prefix: 10. if prefix == (2,) else 0., .5)
        self.assertEqual(ranked[0][0], (2,))
        self.assertEqual({p for p, _ in ranked}, {p for p, _ in candidates})


if __name__ == '__main__':
    unittest.main()
