# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

"""Functional reward implementations."""

import logging
import re
from abc import abstractmethod
import math
from typing import MutableMapping, Literal

from pydantic import BaseModel
import numpy as np
import torch

log = logging.getLogger(__name__)

from compose_rl.algorithms.reward_modeling.base_reward import Reward, Tokenizer
from compose_rl.utils.rlvr_utils import (
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
    extract_and_build_pydantic_object,
)


class IncreasingNumbersReward(Reward):
    """Reward based on the number of generated increasing numbers.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for the reward.
    """

    # This can be run async
    BLOCKING = False

    def __init__(self, tokenizer: Tokenizer):
        super().__init__(tokenizer=tokenizer)

    @staticmethod
    def is_number(text: str):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Creates a reward based on the number of generated increasing numbers.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the increasing numbers reward.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        assert 'zero_rewards' in batch.keys()
        assert 'raw_untokenized_texts' in batch.keys()
        assert 'generated_lens' in batch.keys()

        rewards = batch['zero_rewards']
        raw_untokenized_texts = batch['raw_untokenized_texts']
        generated_lens = batch['generated_lens']

        batch_size = rewards.shape[0]
        all_generated_texts = [x[1] for x in raw_untokenized_texts]
        curr_rewards = []
        for gen_text in all_generated_texts:
            gen_tokens = gen_text.split()
            number_tokens = [
                float(token)
                for token in gen_tokens
                if IncreasingNumbersReward.is_number(token)
            ]
            if len(number_tokens) > 0:
                sorted_count = 1
                previous_token = number_tokens[0]
                for token in number_tokens[1:]:
                    if token > previous_token:
                        sorted_count += 1
                        previous_token = token
                    else:
                        break
                curr_rewards.append((sorted_count) / max(len(gen_tokens), 1))
            else:
                curr_rewards.append(0)
        curr_rewards = torch.tensor(curr_rewards).to(
            rewards.device,
        ).type(rewards.dtype)
        rewards[torch.arange(batch_size), generated_lens - 1] += curr_rewards
        return rewards


class ShortResponseReward(Reward):
    """Reward based on the length of the generated response.

    Args:
        reward (float): The reward to apply.
        len_threshold (int): The length threshold to apply the reward.
        tokenizer (Tokenizer): The tokenizer to use for the reward.
    """

    # This can be run async
    BLOCKING = False

    def __init__(self, reward: float, len_threshold: int, tokenizer: Tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.reward = reward
        self.len_threshold = len_threshold

        log.info(
            f'Adding a reward of {self.reward} if a model generates ' +
            f'tokens under the length {self.len_threshold}',
        )

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Apply the reward to the EOS tokens and nothing else.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the short response reward.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        assert 'zero_rewards' in batch.keys()
        assert 'generated_lens' in batch.keys()

        rewards = batch['zero_rewards']
        generated_lens = batch['generated_lens']
        bs = generated_lens.size(0)
        for i in range(bs):
            if generated_lens[i] <= self.len_threshold:
                rewards[i, generated_lens[i] - 1] += self.reward
        return rewards


class BadGenerationEndReward(Reward):
    """Reward based on the end of the generated response.

    Args:
        reward (float): The reward to apply.
        eos_penalty (bool): The penalty to apply if the response does not end with an EOS.
        tokenizer (Tokenizer): The tokenizer to use for the reward.
        extra_special_tokens (list[str] | None): The extra special tokens to check for.
            Defaults to `None`.
    """

    # This can be run async
    BLOCKING = False

    def __init__(
        self,
        reward: float,
        eos_penalty: bool,
        tokenizer: Tokenizer,
        extra_special_tokens: list[str] | None = None,
    ):
        super().__init__(tokenizer=tokenizer)
        self.reward = reward
        self.eos_penalty = eos_penalty

        # Extra special tokens for any other formats with pseudo EOS alternatives like ChatML
        self.extra_special_tokens = [
            str(tok) for tok in extra_special_tokens
        ] if extra_special_tokens is not None else []
        self.extra_special_token_ids = []
        if self.extra_special_tokens != []:
            self.extra_special_token_ids.extend([
                tok[0] for tok in self.tokenizer(
                    self.extra_special_tokens,
                )  # pyright: ignore
                ['input_ids']
            ])
        if self.eos_penalty:
            # Because tokenizer can be optional, we need to ignore
            self.extra_special_token_ids.append(
                self.tokenizer.eos_token_id,  # pyright: ignore
            )
        log.info(
            f'Subtracting a reward of {self.reward} if a model does not' +
            f'end with an EOS or given set of special tokens',
        )

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Rewards if the generated sequences don't end in EOS or special token.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the bad generation end reward.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        assert 'zero_rewards' in batch.keys()
        assert 'seq_lens' in batch.keys()
        assert 'input_ids' in batch.keys()
        assert 'generated_lens' in batch.keys()

        rewards = batch['zero_rewards']
        seq_lens = batch['seq_lens']
        input_ids = batch['input_ids']
        generated_lens = batch['generated_lens']

        for i in range(generated_lens.size(0)):
            curr_end_token_id = input_ids[i, seq_lens[i] - 1]
            if curr_end_token_id.item() not in self.extra_special_token_ids:
                rewards[i, generated_lens[i] - 1] += self.reward
        return rewards


class OutputLengthReward(Reward):
    """Reward based on the length of the generated response.

    Args:
        max_gen_len (int): The maximum length of the generated response.
        tokenizer (Tokenizer): The tokenizer to use for the reward.
    """

    # This can be run async
    BLOCKING = False

    def __init__(self, max_gen_len: int, tokenizer: Tokenizer):
        super().__init__(tokenizer=tokenizer)
        self.max_gen_len = max_gen_len

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Rewards based on how many output tokens are generated.

        Args:
            batch (dict): The input batch containing all the information we need to compute
                the output length reward.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        assert 'zero_rewards' in batch.keys()
        assert 'generated_lens' in batch.keys()

        rewards = batch['zero_rewards']
        generated_lens = batch['generated_lens']

        batch_size = rewards.shape[0]
        curr_rewards = generated_lens / self.max_gen_len
        rewards[torch.arange(batch_size), generated_lens - 1] += curr_rewards
        return rewards


class BaseVerifierReward(Reward):
    """Base class for verifier rewards.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for the reward.
        reward (float): The reward to apply. Default is 1.0.
    """

    # This can be run async
    BLOCKING = False

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0):
        super().__init__(tokenizer=tokenizer)
        if reward <= 0.0:
            raise ValueError(
                f'Reward for verifiers must be positive, but got {reward}',
            )

        self.reward = reward
        log.info(
            f'Using reward value of {self.reward} for {self.__class__.__name__} verifier',
        )

    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.Tensor:
        """Apply the reward for verifying the correct answer from the model.

        Currently verifier rewards are only applied to the last token of the sequence.

        Args:
            batch (dict): The input batch containing all information needed.

        Returns:
            torch.tensor: rewards of shape <batch_size, seq_len>
        """
        try:
            assert 'zero_rewards' in batch.keys()
            assert 'raw_untokenized_texts' in batch.keys()
            assert 'verified_answers' in batch.keys()
            assert 'generated_lens' in batch.keys()
        except AssertionError as e:
            log.error(f'Missing key in reward batch. Batch keys: {batch.keys()}. Error: {e}')
            raise e
        
        rewards = batch['zero_rewards']
        raw_untokenized_texts = batch['raw_untokenized_texts']
        verified_answers = batch['verified_answers']
        generated_lens = batch['generated_lens']

        batch_size = rewards.shape[0]
        all_generated_texts = [x[1] for x in raw_untokenized_texts]
        for i in range(batch_size):
            # Process based on verifier type
            if self.needs_extraction():
                _answer = self.extract_solution(all_generated_texts[i])
                _reward = self.score_generations(_answer, verified_answers[i])
            else:
                # Score directly without extraction
                _reward = self.score_generations(
                    all_generated_texts[i],
                    verified_answers[i],
                )

            rewards[i, generated_lens[i] - 1] += _reward
        return rewards

    def needs_extraction(self) -> bool:
        """Determine if this verifier needs to extract solutions before scoring.

        Override in child classes if needed.

        Returns:
            bool: True if extraction is needed, False otherwise.
        """
        return True

    def extract_solution(self, text: str) -> str:
        """Extract the solution from text.

        Default implementation raises error; override in child classes if needed.

        Args:
            text (str): The generated text.

        Returns:
            str: The extracted solution.
        """
        raise NotImplementedError(
            'Subclasses must implement `extract_solution` if `needs_extraction` returns True.',
        )

    @abstractmethod
    def score_generations(self, answer: str, label: str) -> float:
        """Score the generated answer against the label.

        Args:
            answer (str): The extracted answer.
            label (str): The verified answer.

        Returns:
            float: The reward score.
        """
        raise NotImplementedError(
            'Subclasses must implement `score_generations` definition.',
        )


class GSM8KVeriferReward(BaseVerifierReward):

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0):
        super().__init__(tokenizer=tokenizer, reward=reward)

    def extract_solution(self, text: str) -> str:
        """Extract numerical solution from GSM8K-style responses."""
        numbers = re.findall(r'-?[\d,]*\.?\d+', text)
        final_answer = ''
        if len(numbers) > 0:
            final_answer = numbers[-1].strip().lower().replace(',', '').replace(
                '$',
                '',
            )

        return final_answer

    def score_generations(self, answer: str, label: str) -> float:
        """Score based on exact match."""
        return self.reward if answer == label else 0.0


class GSM8KFormatVeriferReward(BaseVerifierReward):

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0):
        super().__init__(tokenizer=tokenizer, reward=reward)

    def needs_extraction(self) -> bool:
        """Indicate that this verifier doesn't need extraction."""
        return False

    def score_generations(self, answer: str, label: str) -> float:
        """Check if the answer follows the format with '####' marker.

        Note: The label parameter is not used in this implementation but is required
        by the interface.
        """
        solution = re.search(r'####.*?([\d,]+(?:\.\d+)?)', answer)
        return self.reward if solution is not None else 0.0


class MATHVerifierReward(BaseVerifierReward):

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0):
        super().__init__(tokenizer=tokenizer, reward=reward)

    def extract_solution(self, text: str) -> str:
        """Extract numerical solution from MATH-style responses."""
        last_boxed_string = last_boxed_only_string(text)
        if not last_boxed_string:
            # No boxed string found, so we can't evaluate
            return ''

        unnormalized_answer = remove_boxed(last_boxed_string)
        return normalize_final_answer(unnormalized_answer)

    def score_generations(self, answer: str, label: str) -> float:
        """Score based on exact match or sympy equivalence checks."""
        if answer.strip() == label.strip() or is_equiv(answer, label):
            return self.reward
        return 0.0


class MATHFormatVerifierReward(BaseVerifierReward):

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0):
        super().__init__(tokenizer=tokenizer, reward=reward)

    def needs_extraction(self) -> bool:
        """Indicate that this verifier doesn't need extraction."""
        return False

    def score_generations(self, answer: str, label: str) -> float:
        r"""Check if the answer follows the format with '\\boxed{{}}' marker.

        Note: The label parameter is not used in this implementation but is required
        by the interface.
        """
        last_boxed_string = last_boxed_only_string(answer)
        return 0.0 if not last_boxed_string else self.reward


class MCQAVerifierReward(BaseVerifierReward):

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0):
        super().__init__(tokenizer=tokenizer, reward=reward)
        # Logic borrowed from here: https://github.com/NousResearch/atropos/blob/6386a5e18517d19546a3a77943cbc5081f197ca2/environments/mcqa_thinking_env.py#L208-L300
        # and modified to be efficient via regex patterns
        self._ANSWER_EXACT = re.compile(
            r"""
            (?:(?<=\n)|^)                       # start of string or new-line
            (?:                                 # label / verb phrase variants
                (?:final|correct|best|exact|
                    true|only|real)?\s*answer   # "… answer"
                (?:\s+is)?                      # optional "is"
            | the\s+(?:final\s+)?answer\s+is    # "the answer is…"
            | thus\s*,?\s*final\s+answer        # "thus, final answer…"
            )
            \s*[:=\-–—]?\s*                     # delimiter (:, =, –, —, -)
            (?:\\boxed\s*)?                     # optional LaTeX \boxed
            [\*\(\{\[]?                         # optional opening wrapper
            (?P<ans>[A-Z])                      # ← captured letter (A–Z, case-insens.)
            [\]\}\)\*]?                         # optional closing wrapper
            (?:\s*[)\].]*)?                     # trailing ) . ] …
            """,
            re.IGNORECASE | re.VERBOSE | re.DOTALL,
        )
        self._FALLBACK = re.compile(r'(?<![A-Z])([A-Z])(?![A-Z])')

    def extract_solution(self, text: str) -> str:
        """Extract string answer from responses."""
        if (m := self._ANSWER_EXACT.search(text)):
            return m.group('ans').upper()

        candidates = self._FALLBACK.findall(text.upper())
        return candidates[-1] if candidates else ''

    def score_generations(self, answer: str, label: str) -> float:
        """Score based on exact match."""
        return self.reward if answer == label else 0.0

class ThinkingFormatVerifierReward(BaseVerifierReward):

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0,
                 thinking_tag: str = 'thinking'):
        super().__init__(tokenizer=tokenizer, reward=reward)
        self.thinking_tag = thinking_tag

    def needs_extraction(self) -> bool:
        """Indicate that this verifier doesn't need extraction."""
        return False
    
    def score_generations(self, answer: str, label: str) -> float:
        """
        Checks if the response string contains an opening and closing thinking tag.

        Args:
            answer (str): The response string to check.

        Returns:
            bool: True if the response string contains an opening and closing
             XML tag with the thinking tag, False otherwise.
        """
        if f'<{self.thinking_tag}>' in answer.lower() and \
            f'</{self.thinking_tag}>' in answer.lower():
            return self.reward
        return 0.0
    
class PydanticFormatVerifierReward(BaseVerifierReward):
    """
    Verifier that checks if the last complete JSON object in the response string
    can be parsed into a valid instance of the Pydantic class.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for the reward.
        reward (float): The reward value to return if response is valid.
        pydantic_class (BaseModel): The Pydantic class to parse the response string into.
    """

    def __init__(self, tokenizer: Tokenizer, reward: float,
                 pydantic_class: BaseModel):
        super().__init__(tokenizer=tokenizer, reward=reward)
        self.pydantic_class = pydantic_class

    def needs_extraction(self) -> bool:
        """Indicate that this verifier doesn't need extraction."""
        return False
    
    def score_generations(self, answer: str, label: str) -> float:
        """
        Checks if the response string contains a valid JSON object that can be
        parsed into a valid instance of the Pydantic class.

        Args:
            answer (str): The response string to check.

        Returns:
            bool: True if the response string is a valid instance of the Pydantic class, False otherwise.
        """
        pydantic_obj = extract_and_build_pydantic_object(answer, self.pydantic_class)
        if pydantic_obj is None:
            return 0.0
        return self.reward
    
class Judgement(BaseModel):
    rationale: str
    score: float
    
class JudgementClassification(BaseModel):
    rationale: str
    result: Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]

class JudgementFormatVerifierReward(PydanticFormatVerifierReward):
    """
    Verifier that checks if the last complete JSON object in the response string
    can be parsed into a valid instance of the Judgement class.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for the reward.
        reward (float): The reward value to return if response is valid.
    """

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0):
        super().__init__(tokenizer=tokenizer, reward=reward,
                         pydantic_class=Judgement)
    

class JudgementScoreVerifierReward(BaseVerifierReward):

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0, score_is_probability: bool = False, class_mode: bool = False):
        super().__init__(tokenizer=tokenizer, reward=reward)
        self.score_is_probability = score_is_probability
        self.class_mode = class_mode
        if self.score_is_probability and self.class_mode:
            raise ValueError(
                'Cannot use score_is_probability with class_model. ' +
                'Use one or the other.',
            )
        # Used in class mode
        self._logit_map = {
            k: float(v)
            for k, v in zip(
                'ABCDEFGHIJKLMNOPQRST',
                np.linspace(-7.0, 7.0, 20),
            )
        }

    def needs_extraction(self) -> bool:
        """Indicate that this verifier needs extraction."""
        return True
    
    def extract_solution(self, text: str) -> float | None:
        """Extract the score from text, if possible and valid."""
        judgement_obj = extract_and_build_pydantic_object(text, Judgement if not self.class_mode else JudgementClassification)
        if judgement_obj is None:
            return None
        if self.class_mode:
            return self._logit_map.get(judgement_obj.result, None)
        return judgement_obj.score
    
    def score_generations(self, answer: float | None, label: bool | int) -> float:
        """
        Rewards the model for generating a valid response with a score that is 
        similar to the ground truth score.

        Args:
            answer (float): The logit of the generated response.
            label (bool): The boolean class of the ground truth response. (Yes or No)
        
        Returns:
            float: The reward value. A value between [0, self.reward]]
        """
        if answer is None:
            # we likely could not extract a score from the response. so fail.
            return 0.0
        if not np.isfinite(answer):
            # if the answer is NaN, our reward would be NaN (bad), so let's avoid this, shall we?
            # seems smart to avoid infinite values as well
            return 0.0
        # if the target is an integer, we need to convert it to a boolean
        if isinstance(label, int) and label in [0, 1]:
            label = bool(label)
        reward_scalar = self._mse_reward(answer, label)
        return self.reward * reward_scalar
    
    def _safe_sigmoid(self, x, clip_min=-250, clip_max=250):
        if x > clip_max:
            return 1.0
        if x < clip_min:
            return 0.0
        return 1.0 / (1.0 + np.exp(-x))


    def _mse_reward(self, generated_score: float, target: bool) -> float:
        """
        Returns 1 minus the squared error between the generated probability and the target.

        This is then used to scale the reward output. The further generated_score
        is from the target, the more downscaled the reward will be.

        Args:
            generated_score (float): The logit of the generated response. Or, if the flag
                `score_is_probability` is set to True, the probability.
            target (bool): The boolean class of the ground truth response. (Yes or No)

        Returns:
            float: [0, 1]
        """
        assert isinstance(target, bool), f'Target must be a boolean, got {type(target)}'
        target = 1.0 if target else 0.0
        if not self.score_is_probability:
            generated_probability = self._safe_sigmoid(generated_score)
        else:
            generated_probability = generated_score
        if not (0.0 <= generated_probability <= 1.0):
            # If the generated probability is not in [0, 1], we cannot compute a valid reward.
            # This is effectively treated as a formatting failure (0 reward).
            return 0.0
        return 1.0 - (generated_probability - target) ** 2
    
# Helper class for reward to follow
class DigitEntropyScorer:
    """
    A class to score sequences of digits based on their likelihood given previous digits.
    
    This class uses a recursive dictionary structure to store digit frequencies at each position
    after the decimal point. It allows for efficient scoring of sequences by calculating the
    unlikelihood (?? 1 - likelihood) of each digit given the previous digits.
    
    This is a way to reward precise sequences while promoting diversity in the sequences.
    """
    def __init__(self, depth: int=8, gamma: float=1.0):
        self.depth = depth
        self.gamma = gamma
        if not (0 < self.gamma <= 1):
            raise ValueError(f'Gamma must be in (0, 1], got {self.gamma}')
        self.count = 0
        self.children: dict[str, DigitEntropyScorer] = {}
        
        weights = 0.8 ** (np.arange(self.depth) + 1)  # Exponential decay for each position
        self.weights = weights/weights.sum()
        
    def _decay(self):
        """
        Decay the counts of all children by gamma.
        This is used to keep the probabilities up-to-date as new sequences are added.
        """
        if self.gamma == 1.0:
            return
        self.count *= self.gamma
        for child in self.children.values():
            child._decay()

    def _add(self, digits: str, pos: int=0):
        """
        Add a sequence of digits (as a string) to the dictionary, updating counts recursively.
        """
        self.count += 1
        if pos < len(digits) and pos < self.depth:
            digit = digits[pos]
            if digit not in self.children:
                self.children[digit] = DigitEntropyScorer(self.depth, self.gamma)
            self.children[digit]._add(digits, pos + 1)

    def _get_freq(self, digits: str, pos: int=0):
        """
        Get the frequency of the digit at position pos given the previous digits.
        Returns the probability of the digit at pos, given the prefix digits[:pos].
        """
        if pos == len(digits) or pos == self.depth or self.count == 0:
            return 0.0
        digit = digits[pos]
        if digit in self.children:
            return self.children[digit].count / (1e-10 + self.count)
        else:
            return 0.0
        
    def _get_frequencies(self, digits: str):
        """
        Get the conditional frequencies of the digits in the sequence
        """
        frequencies = []
        node = self
        for pos, digit in enumerate(digits[:self.depth]):
            if node.count == 0:
                frequencies.append(0.0)
            else:
                freq = node._get_freq(digits, pos)
                frequencies.append(freq)
            node = node.children.get(digit, DigitEntropyScorer(self.depth, self.gamma))
        return np.array(frequencies)

    def _get_score(self, digits: str):
        """
        For a sequence of digits, sum the unlikelihood for each digit position.
        """
        scores = 1 - self._get_frequencies(digits)
        return np.sum(scores * self.weights[:len(scores)])  # Apply weights to the scores
    
    def score(self, number: float) -> float:
        """
        Get the score for a number based on its digits.
        """
        digits = str(number).split('.')[-1]
        return self._get_score(digits)
    
    def add_and_score(self, number: float) -> float:
        """
        Add a number to the dictionary and return the score for its digits.
        """
        self._decay()  # Decay counts before adding new number
        digits = str(number).split('.')[-1]
        score = self._get_score(digits)
        self._add(digits)
        return score
    
# Note: We need to hack the reward manager if we want to use this reward.
# Specifically, we need to prevent the reward manager from trying to do async calls
# because it prevents this class from maintaining its internal state.
class JudgmentLogitDiversityReward(Reward):
    """
    A reward that encourages diversity in generated sequences by scoring based on the
    unlikelihood of digits given previous digits.
    
    Args:
        tokenizer (Tokenizer): The tokenizer to use for the reward.
        reward (float): The base reward value to apply.
        depth (int): The depth of the digit sequence to consider for scoring.
        gamma (float): The decay factor for the digit frequency counts.
            Should be in the range (0, 1]. Defaults to 1.0.
        ones_place_weighting (float): An additional scorer will follow the pre-decimal digit.
            (The number will be clipped to [-9, 9] for this.) This argument sets the weight contribution,
            from 0 (no weight) to 1 (full weight).
            Defaults to 0.0, meaning the pre-decimal digits are ignored.
     """
    
    BLOCKING = False

    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0, depth: int=8, gamma: float=1.0, ones_place_weighting: float = 0.0):
        super().__init__(tokenizer=tokenizer)
        self.reward = reward
        self.scorer = DigitEntropyScorer(depth=depth, gamma=gamma)
        self.ones_place_weighting = ones_place_weighting
        if not (0 <= self.ones_place_weighting <= 1):
            raise ValueError(
                f'Ones place weighting must be in [0, 1], got {self.ones_place_weighting}',
            )
        if self.ones_place_weighting > 0:
            # We need a separate scorer for the ones place
            self.ones_place_scorer = DigitEntropyScorer(depth=depth, gamma=gamma)
        else:
            self.ones_place_scorer = None
    
    def __call__(
        self,
        batch: MutableMapping,
    ) -> torch.FloatTensor:
        try:
            assert 'zero_rewards' in batch.keys()
            assert 'raw_untokenized_texts' in batch.keys()
            assert 'generated_lens' in batch.keys()
        except AssertionError as e:
            log.error(f'Missing key in reward batch. Batch keys: {batch.keys()}. Error: {e}')
            raise e
        
        rewards = batch['zero_rewards']
        raw_untokenized_texts = batch['raw_untokenized_texts']
        generated_lens = batch['generated_lens']

        batch_size = rewards.shape[0]
        all_generated_texts = [x[1] for x in raw_untokenized_texts]
        for i in range(batch_size):
            rewards[i, generated_lens[i] - 1] += self._compute_reward(all_generated_texts[i])
        print(f'JudgmentLogitDiversityReward count: {self.scorer.count}')
        return rewards
    
    def _compute_reward(self, text: str) -> float:
        """Extract the score from text, if possible and valid."""
        judgement_obj: Judgement | None = extract_and_build_pydantic_object(text, Judgement)
        if judgement_obj is None:
            # We could not extract a score from the response. so fail.
            return 0.0
        score = judgement_obj.score
        
        if not np.isfinite(score):
            # If the answer is NaN, our reward would be NaN (bad), so let's avoid this, shall we?
            # seems smart to avoid infinite values as well
            return 0.0
        main_reward = self.scorer.add_and_score(float(score))
        
        if self.ones_place_scorer is None:
            return self.reward * main_reward
        
        # If we are rewarding the ones place, we'll do some hocus pocus to get this into a number
        # that works nice for the scorer.
        # Basically, the below will turn a number like 3.5323 into 0.3, or -8.1234 into -0.8. Nothing too crazy.
        num = float(str(float(np.abs(score).clip(-9, 9)/10))[:3])
        ones_place_reward = self.ones_place_scorer.add_and_score(num)
        
        net_reward = ((1 - self.ones_place_weighting) * main_reward) + (self.ones_place_weighting * ones_place_reward)
        return self.reward * net_reward

# Example system prompt for the JudgmentOmniReward class (see class below for details):
"""You will be asked to perform a judgment resulting in a yes/no decision.

The message to follow will provide content to be judged and instructions on what aspects to consider in your judgment.

Those instructions will call for a JSON output with a "rationale" and "result" field.
Instead of following those instructions specifically, you should change 2 aspects.
1. You must begin all your responses with a private thinking section wrapped in <thinking> xml tags.
2. Instead of answering "yes" or "no" in the "result" field, you should output a floating-point score indicating your confidence level in the judgment. This will be interpreted as a classification logit, so 0.0 means totally uncertain, negative values mean "no" and positive values mean "yes". More extreme values indicate more confidence in your assessment.

Importantly, your <thinking> must systematically derive the score you output. You must do this by structuring your thinking as follows:
1. Identify at least 3 possible reasons why the score might be yes. For each such reason, assign a POSITIVE score indicating the marginal contribution of that reason to the overall score. Place each positive reason+score in a <pro> tag.
2. Identify at least 3 negative reasons why the score might be no, but assign a NEGATIVE score indicating the marginal contribution of that reason to the overall score. Place each negative reason+score in a <con> tag.
3. All marginal scores should be precise -- report at least 4 decimal places.
4. The final score you output should be the sum of all the marginal scores. Note that you should be summing at least 6 scores!

Notes on the <thinking> formatting rules:
- You are encouraged to think out loud in whatever order you like. You can adjudicate the pros and cons in any order, and you can even change your mind as you go.
- Every time you want to increment or decrement the score, open a <pro> or <con> tag, respectively, and then close it when you are done with that reason.
- You must nest a <score> section inside each <pro> or <con> tag, and this must contain a single floating-point number with at least 4 decimal places. This is the marginal score for that reason.
- <score> values must be positive within <pro> tags and negative within <con> tags.
- The thinking section will be considered incomplete if there are any fewer than 3 <pro> tags and 3 <con> tags. But you should use even more than that if you can!!!
- Because this formatting can be verified, your response will be considered a failure if you do not adhere.

To clarify, here are 2 examples of valid <thinking> structures:
<thinking> # just an example
[A long, detailed thought process that considers multiple aspects of the judgment, including potential uncertainties and complexities in the situation.]
Aspect X
<pro>[A reason why the score might be yes]<score>[the marginal (positive) score associated with this reason]</score></pro>
<con>[A reason why the score might be no]<score>[the marginal (negative) score associated with this reason]</score></con>
Aspect Y
[similar pattern to above, but for a different aspect]
Aspect Z
[similar pattern to above, but for a third aspect]
Additional considerations
<pro>...</pro>
<pro>...</pro>
<con>[An additional point that requires us to reconsider a previous pro and change the running score]</con>

Score tallying
[quick arithmetic to sum the marginal scores and arrive at a final score]
</thinking>

<thinking> # another example
<pro>[A reason why the score might be yes]<score>[the marginal (positive) score associated with this reason]</score></pro>
<pro>...</pro>
<pro>...</pro>
<pro>...</pro>
<pro>...</pro>
<pro>...</pro> 
<con>[A reason why the score might be no]<score>[the marginal (negative) score associated with this reason]</score></con>
<con>...</con>
<con>...</con>

Score tallying
[quick arithmetic to sum the marginal scores and arrive at a final score]
</thinking>

Let's summarize!
You should partially disregard the formatting instructions in the message to follow, and ensure that your output looks like this:

<thinking>
[Think out loud here -- be rigorous and thorough! Follow the structure directions above, and be sure to assign scores to each reason you identify. Your final score should be the sum of all the marginal scores.]
</thinking>
{
    "rationale": "Summarise your judgment here, and account for uncertainity",
    "score": [float] # a floating-point score indicating your confidence level in the judgment, where negative values mean "no" and positive values mean "yes"; this must be the sum of the marginal scores articulated in your <thinking>
}

Hint: the score is actually a classification logit (meaning `sigmoid(logit) --> probability("yes")`), so an absolute value of >=7 indicates ~maximal~ confidence.
"""

def validate_thinking_structure(generation: str) -> bool:
    """
    Validate that the <thinking> section of the generation follows the expected structure.
    It should contain at least 3 <pro> tags and 3 <con> tags, each with a <score> tag inside.
    The <score> tags should contain a floating-point number which must be finite and
    positive/negative depending on whether it's a pro or con.
    """
    # A nest dictionary where each key is a tag name and the value is its contents (or another xml dictionary if the tag contains tags)
    thinking_dict = xml_string_to_dict(generation)
    if 'thinking' not in thinking_dict:
        return False
    thinking_content = thinking_dict['thinking']
    if not isinstance(thinking_content, dict):
        return False  # Thinking content should be a dictionary
    if 'pro' not in thinking_content or 'con' not in thinking_content:
        return False
    pro = thinking_content['pro']
    con = thinking_content['con']
    if not isinstance(pro, list) or not isinstance(con, list):
        return False  # pro and con should be lists of dictionaries
    if len(pro) < 3 or len(con) < 3:
        return False
    # Check that each pro and con has a score
    def validate_pro_con(content: str | dict, is_pro: bool) -> bool:
        if not isinstance(content, dict):
            return False
        if 'score' not in content:
            return False
        try:
            score = float(content['score'])
        except (ValueError, TypeError):
            return False
        if not np.isfinite(score):
            return False
        if (is_pro and score <= 0) or (not is_pro and score >= 0):
            return False
        return True
    for p in pro:
        if not validate_pro_con(p, is_pro=True):
            return False
    for c in con:
        if not validate_pro_con(c, is_pro=False):
            return False
    return True

def extract_total_score(generation: str) -> float:
    """
    Extract the total score from the generation's <thinking> section.
    This assumes that `validate_thinking_structure` has already been called and passed.
    """
    # A nest dictionary where each key is a tag name and the value is its contents (or another xml dictionary if the tag contains tags)
    thinking_dict = xml_string_to_dict(generation)
    score = 0.0
    for pro in thinking_dict['thinking']['pro']:
        score += float(pro['score'])
    for con in thinking_dict['thinking']['con']:
        score += float(con['score'])
    return score
    
class JudgmentOmniReward(BaseVerifierReward):
    """A single reward for handling the thinking and scoring side of judging.
    
    Note: This enforces some formatting rules, which match a system prompt for this task.
      An example of such a system prompt can be found in the comments above the source code
      for this class.
      
    The formatting rules are:
    - The response must start with a <thinking> section.
    - The <thinking> section must contain at least 3 <pro> tags and 3 <con> tags.
    - Each <pro> tag must contain a <score> tag with a positive floating-point number.
    - Each <con> tag must contain a <score> tag with a negative floating-point number.
    
    In addition, the final JSON output must contain a "rationale" field and a "score" field.
    And the "score" value must be the sum of all the scores in the <thinking> section.
    
    The reward structure is:
    - If the generation does not follow the formatting rules, the reward is 0.0.
    - Passing the formatting rules adds 0.1 to the reward.
    - From there, add to the reward: 0.9 * (1 - the MSE between `sigmoid(score)` and the target label). _
    """
    def __init__(self, tokenizer: Tokenizer, reward: float = 1.0, max_score_disagreement: float = 1e-4):
        super().__init__(tokenizer=tokenizer, reward=reward)
        self.max_score_disagreement = max_score_disagreement
        if self.max_score_disagreement <= 0:
            raise ValueError(
                f'max_score_disagreement must be positive, got {self.max_score_disagreement}',
            )

    def needs_extraction(self) -> bool:
        """Indicate that this verifier needs extraction."""
        return False
    
    def score_generations(self, answer: str, label: bool | int) -> float:
        """
        Rewards the model for generating a valid response with a score that is 
        similar to the ground truth score.

        Args:
            answer (str): The text produced by the model.
            label (bool): The boolean class of the ground truth response. (Yes or No)
        
        Returns:
            float: The reward value. A value between [0, self.reward]]
        """
        # First, validate the thinking structure
        if not validate_thinking_structure(answer):
            # If the thinking structure is invalid, we return 0.0 reward
            return 0.0
        # If the thinking structure is valid, we can extract the score
        thinking_score = extract_total_score(answer)
        if not np.isfinite(thinking_score):
            # If the score is NaN, our reward would be NaN (bad), so let's avoid this, shall we?
            # seems smart to avoid infinite values as well
            return 0.0
        
        # Now let's parse the JSON output
        judgement_obj = extract_and_build_pydantic_object(answer, Judgement)
        if judgement_obj is None:
            # We could not extract a score from the response. so fail.
            return 0.0
        # Our final formatting check:
        if np.abs(judgement_obj.score - thinking_score) > self.max_score_disagreement:
            return 0.0
        
        base_reward = 0.1  # Base reward for passing the formatting rules
        
        # if the target is an integer, we need to convert it to a boolean
        if isinstance(label, int) and label in [0, 1]:
            label = bool(label)
        mse_reward = 0.9 * self._mse_reward(judgement_obj.score, label)
        
        reward = base_reward + mse_reward
        
        return self.reward * reward
    
    def _safe_sigmoid(self, x, clip_min=-250, clip_max=250):
        if x > clip_max:
            return 1.0
        if x < clip_min:
            return 0.0
        return 1.0 / (1.0 + np.exp(-x))

    def _mse_reward(self, generated_score: float, target: bool) -> float:
        """
        Returns 1 minus the squared error between the generated probability and the target.

        This is then used to scale the reward output. The further generated_score
        is from the target, the more downscaled the reward will be.

        Args:
            generated_score (float): The logit of the generated response. Or, if the flag
                `score_is_probability` is set to True, the probability.
            target (bool): The boolean class of the ground truth response. (Yes or No)

        Returns:
            float: [0, 1]
        """
        assert isinstance(target, bool), f'Target must be a boolean, got {type(target)}'
        target = 1.0 if target else 0.0
        generated_probability = self._safe_sigmoid(generated_score)
        if not (0.0 <= generated_probability <= 1.0):
            # If the generated probability is not in [0, 1], we cannot compute a valid reward.
            # This is effectively treated as a formatting failure (0 reward).
            return 0.0
        return 1.0 - (generated_probability - target) ** 2
    
# Helper for parsing XML strings into dictionaries
def xml_string_to_dict(xml_string: str, allow_multiple: bool = True) -> dict:
    """
    Recursively parses an XML string into a dictionary.

    Args:
        xml_string (str): The XML string to parse.
        allow_multiple (bool, optional): Whether to allow multiple values for the same tag. Defaults to True. 
          If this is false, and there are multiple tags with the same name, an error will be raised.
    Returns:
        dict: A dictionary representing the parsed XML.
    """
    # Regular expression pattern to match XML tags
    pattern = r'<(.*?)>(.*?)</\1>'
    
    # Find all matches of the pattern in the XML string
    matches = re.findall(pattern, xml_string, flags=re.IGNORECASE | re.DOTALL)
    
    # Initialize an empty dictionary to store the results
    result: dict[str, str | list[str | dict]] = {}
    
    # Iterate over the matches
    for match in matches:
        # The first element of the match is the tag name
        tag_name = match[0]
        
        # The second element of the match is the tag contents
        tag_contents = match[1]
        
        # If the tag contents contain XML tags, recursively parse them
        submatch = re.findall(pattern, tag_contents, flags=re.IGNORECASE | re.DOTALL)
        # Check if there are nested XML tags
        if submatch:
            # Recursively parse nested XML
            value = xml_string_to_dict(tag_contents, allow_multiple)
        else:
            # If no nested tags, just use the stripped content
            value = tag_contents.strip()
        
        # Check if the tag name already exists in the result dictionary
        if tag_name in result:
            if allow_multiple:
                # If multiple values are allowed, convert to list or append
                if isinstance(result[tag_name], list):
                    result[tag_name].append(value) # type: ignore
                else:
                    result[tag_name] = [result[tag_name], value] # type: ignore
            else:
                # If multiple values are not allowed, raise an error
                raise ValueError(f"Multiple occurrences of tag '{tag_name}' found, but allow_multiple is False.")
        else:
            # If tag name doesn't exist, add it to the result dictionary
            result[tag_name] = value # type: ignore
    # Return the result dictionary 
    return result