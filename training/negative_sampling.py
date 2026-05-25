from __future__ import annotations

from typing import Set, Tuple

import torch

Triple = Tuple[int, int, int]


class NegativeSampler:
    def __init__(
        self,
        num_entities: int,
        true_triples: Set[Triple],
        negative_samples: int,
        seed: int,
    ) -> None:
        self.num_entities = num_entities
        self.true_triples = true_triples
        self.negative_samples = negative_samples
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def sample(self, positives: torch.LongTensor) -> torch.LongTensor:
        batch_size = positives.shape[0]
        negatives = positives[:, None, :].repeat(1, self.negative_samples, 1).clone()
        replace_heads = torch.rand(
            batch_size,
            self.negative_samples,
            generator=self.generator,
        ) < 0.5
        replacements = torch.randint(
            low=0,
            high=self.num_entities,
            size=(batch_size, self.negative_samples),
            generator=self.generator,
            dtype=torch.long,
        )
        negatives[:, :, 0] = torch.where(replace_heads, replacements, negatives[:, :, 0])
        negatives[:, :, 2] = torch.where(~replace_heads, replacements, negatives[:, :, 2])

        flat = negatives.view(-1, 3)
        for idx in range(flat.shape[0]):
            attempts = 0
            while tuple(int(v) for v in flat[idx].tolist()) in self.true_triples and attempts < 20:
                if replace_heads.view(-1)[idx]:
                    flat[idx, 0] = torch.randint(
                        self.num_entities,
                        size=(1,),
                        generator=self.generator,
                    )
                else:
                    flat[idx, 2] = torch.randint(
                        self.num_entities,
                        size=(1,),
                        generator=self.generator,
                    )
                attempts += 1
        return negatives

