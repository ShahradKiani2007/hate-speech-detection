import os
import time
import random
import json
from pathlib import Path
from typing import List

import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from dotenv import load_dotenv
 
load_dotenv()


class HateSpeechClassification(BaseModel):
    label_id: int = Field(
        description="Class index: 0 for Hate Speech, 1 for Offensive Language, 2 for Neither"
    )
    label_name: str = Field(
        description="Class name: 'Hate Speech', 'Offensive Language', or 'Neither'"
    )


class SingleItemResult(BaseModel):
    index: int = Field(description="The tweet number this classification refers to, matching the input numbering")
    label_id: int = Field(description="Class index: 0 for Hate Speech, 1 for Offensive Language, 2 for Neither")
    label_name: str = Field(description="Class name: 'Hate Speech', 'Offensive Language', or 'Neither'")


class GroupClassification(BaseModel):
    results: List[SingleItemResult] = Field(
        description="One classification result per input tweet, in the same order, "
        "with 'index' matching the tweet's number in the input list"
    )


# ---------------------------------------------------------------------------
# Trimmed to 3 few-shot examples (one per class) to cut prompt tokens. This
# is a tradeoff: fewer examples means slightly less coverage of ambiguous
# cases, but with 3718 tweets and a 500/day call budget, keeping the prompt
# lean matters more here than squeezing out the last bit of accuracy.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "tweet": "These immigrants are ruining this country, they should all be sent back where they came from",
        "label_id": 0,
        "label_name": "Hate Speech",
    },
    {
        "tweet": "my bitches already know what it is, we don't play about each other",
        "label_id": 1,
        "label_name": "Offensive Language",
    },
    {
        "tweet": "I can't believe the referee made that call, absolute robbery",
        "label_id": 2,
        "label_name": "Neither",
    },
]


def _format_few_shot_block(examples):
    blocks = []
    for ex in examples:
        blocks.append(
            f'Tweet: "{ex["tweet"]}"\n'
            f'Classification: {{"label_id": {ex["label_id"]}, "label_name": "{ex["label_name"]}"}}'
        )
    return "\n\n".join(blocks)


# Shortened research-context note -- kept brief on purpose to save tokens,
# but still enough to frame this as a legitimate academic labeling task
# rather than content generation, which helps reduce false-positive blocks.
_RESEARCH_NOTE = (
    "Academic NLP research task: label tweets from a public hate-speech "
    "benchmark dataset for a content-moderation classifier. You are only "
    "labeling existing public text, not generating or endorsing it."
)

_CATEGORY_DEFS = """0: Hate Speech - attacks/dehumanizes/incites violence against a protected group (race, ethnicity, religion, gender, sexual orientation, disability, national origin, etc.)
1: Offensive Language - profanity/slurs/vulgarity NOT targeting a protected group (casual insults, in-group slang, cursing without group-based hatred)
2: Neither - neutral or benign text, no profanity or targeted attack

Note: slurs/profanity used casually or in-group (e.g. reclaimed terms) are usually Offensive, not Hate Speech, unless attacking a protected group."""


def create_prompt(parser):
    few_shot_block = _format_few_shot_block(FEW_SHOT_EXAMPLES)

    prompt = PromptTemplate(
        template=_RESEARCH_NOTE + """

Classify the tweet into exactly one category:

""" + _CATEGORY_DEFS + """

Examples:

{few_shot_block}

Tweet: "{tweet}"

{format_instructions}
""",
        input_variables=["tweet"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "few_shot_block": few_shot_block,
        },
    )

    return prompt


def create_group_prompt(parser):
    few_shot_block = _format_few_shot_block(FEW_SHOT_EXAMPLES)

    prompt = PromptTemplate(
        template=_RESEARCH_NOTE + """

Classify EACH tweet below into exactly one category (treat every tweet independently):

""" + _CATEGORY_DEFS + """

Examples:

{few_shot_block}

Classify these {n_tweets} tweets, numbered from 0. Return exactly {n_tweets} results,
one per tweet, using the same index numbers:

{numbered_tweets}

{format_instructions}
""",
        input_variables=["numbered_tweets", "n_tweets"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "few_shot_block": few_shot_block,
        },
    )

    return prompt


def _looks_like_content_block(err):
    """
    A Gemini safety-filter block produces an empty response, which fails
    JSON parsing with nothing between "Invalid json output:" and the
    troubleshooting link. A genuinely malformed (non-blocked) generation
    would have actual garbled text there instead. Retrying an identical
    blocked tweet is pointless -- it'll be blocked again every time.
    """
    msg = str(err)
    if "invalid json output" not in msg.lower():
        return False
    after = msg.split(":", 1)[-1]
    body = after.split("For troubleshooting")[0]
    return body.strip() == ""


def _build_llm():
    if "GOOGLE_API_KEY" not in os.environ:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Export it as an environment variable, "
            "do not hardcode it in source."
        )

    safety_settings = {
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        safety_settings=safety_settings,
    )


def make_chain(llm):
    parser = PydanticOutputParser(pydantic_object=HateSpeechClassification)
    prompt = create_prompt(parser)
    return prompt | llm | parser


def make_group_chain(llm):
    parser = PydanticOutputParser(pydantic_object=GroupClassification)
    prompt = create_group_prompt(parser)
    return prompt | llm | parser


def estimate_calls(n_tweets, group_size, daily_limit=500):
    """
    Rough call-count estimator for planning around a daily API limit.
    primary_calls: calls needed if every group succeeds on the first try.
    Doesn't account for retries/fallbacks -- treat the gap to daily_limit
    as your safety margin for those.
    """
    primary_calls = -(-n_tweets // group_size)  # ceil division
    margin = daily_limit - primary_calls
    print(
        f"[estimate] {n_tweets} tweets / group_size={group_size} -> "
        f"{primary_calls} primary calls. Daily limit={daily_limit} -> "
        f"margin of {margin} calls for retries/fallbacks."
    )
    if margin < 0:
        print(
            f"[estimate] WARNING: primary calls alone exceed your daily limit. "
            f"Increase group_size or split the run across multiple days."
        )
    elif margin < primary_calls * 0.2:
        print(
            f"[estimate] WARNING: thin margin (<20% of primary calls). "
            f"If several groups fail and fall back to per-tweet calls, "
            f"you may hit the daily limit before finishing."
        )
    return primary_calls, margin


class LlmModel:
    def __init__(self, max_retries=2, base_delay=1.0, batch_size=20, max_concurrency=5,
                 group_size=1):
        """
        max_retries: retry attempts per item/group on failure. Kept low (2)
            by default to conserve API call budget -- content blocks fail
            fast anyway (see _looks_like_content_block), so retries are only
            spent on genuine transient errors.
        base_delay: base seconds for exponential backoff.
        batch_size: how many groups to send per .batch() call (concurrency unit).
        max_concurrency: max parallel requests within a batch call.
        group_size: tweets packed into a single prompt/API call. This is the
            main lever for reducing total API calls -- see estimate_calls().
        """
        self._llm = _build_llm()
        self.chain = make_chain(self._llm)
        self.group_chain = make_group_chain(self._llm) if group_size > 1 else None
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.group_size = group_size
        self.failed_indices = []
        self.calls_made = 0  # rough running count of API calls actually made

    def _invoke_with_retry(self, tweet_text, fallback_label=2, index=None):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                self.calls_made += 1
                result = self.chain.invoke({"tweet": tweet_text})
                return result.label_id
            except Exception as e:
                last_err = e
                if _looks_like_content_block(e):
                    print(
                        f"[warn] tweet={tweet_text[:50]!r} blocked by safety filter "
                        f"(empty response). Failing fast, no point retrying."
                    )
                    break
                sleep_time = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(
                    f"[warn] classification failed (attempt {attempt + 1}/"
                    f"{self.max_retries}) for tweet={tweet_text[:50]!r}: {e}. "
                    f"Retrying in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)

        print(
            f"[error] giving up for tweet={tweet_text[:50]!r}: {last_err}. "
            f"Falling back to label {fallback_label}. REVIEW MANUALLY."
        )
        if index is not None:
            self.failed_indices.append(index)
        return fallback_label

    def _invoke_group_with_retry(self, tweets, fallback_label=2, start_index=None):
        numbered = "\n".join(f'{i}: "{t}"' for i, t in enumerate(tweets))
        last_err = None

        for attempt in range(self.max_retries):
            try:
                self.calls_made += 1
                result = self.group_chain.invoke({
                    "numbered_tweets": numbered,
                    "n_tweets": len(tweets),
                })
                by_index = {r.index: r.label_id for r in result.results}
                if set(by_index.keys()) != set(range(len(tweets))):
                    raise ValueError(
                        f"expected indices 0..{len(tweets) - 1}, got {sorted(by_index.keys())}"
                    )
                return [by_index[i] for i in range(len(tweets))]
            except Exception as e:
                last_err = e
                if _looks_like_content_block(e):
                    print(
                        f"[warn] group (size={len(tweets)}) blocked by safety filter. "
                        f"Falling back to per-tweet classification for this group "
                        f"(uses up to {len(tweets)} extra calls -- factor this into "
                        f"your daily budget if blocks are frequent)."
                    )
                    break
                sleep_time = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                print(
                    f"[warn] group classification failed (attempt {attempt + 1}/"
                    f"{self.max_retries}, group_size={len(tweets)}): {e}. "
                    f"Retrying in {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)

        print(
            f"[error] giving up on group (size={len(tweets)}): {last_err}. "
            f"Falling back to per-tweet classification for this group."
        )
        return [
            self._invoke_with_retry(
                t, fallback_label=fallback_label,
                index=(start_index + i) if start_index is not None else None,
            )
            for i, t in enumerate(tweets)
        ]

    def predict_one(self, text):
        return self._invoke_with_retry(text)

    @staticmethod
    def _save_checkpoint(checkpoint_path, y_pred, n_total):
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = {"n_total": n_total, "predictions": [None if v is None else int(v) for v in y_pred]}
        with open(tmp_path, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, path)

    @staticmethod
    def _load_checkpoint(checkpoint_path, n_total, verbose=True):
        path = Path(checkpoint_path)
        if not path.exists():
            return None
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            if verbose:
                print(f"[warn] checkpoint at {path} unreadable ({e}), starting fresh.")
            return None
        if payload.get("n_total") != n_total:
            if verbose:
                print(
                    f"[warn] checkpoint size mismatch ({payload.get('n_total')} vs "
                    f"{n_total}), starting fresh. X must match the original run exactly."
                )
            return None
        preds = payload.get("predictions")
        if not isinstance(preds, list) or len(preds) != n_total:
            if verbose:
                print(f"[warn] checkpoint at {path} malformed, starting fresh.")
            return None
        n_done = sum(1 for v in preds if v is not None)
        if verbose:
            print(f"[info] resuming from checkpoint: {n_done}/{n_total} rows already done.")
        return preds

    def predict(self, X, fallback_label=2, verbose=True, checkpoint_path=None, checkpoint_every=1):
        X = list(X)
        if self.group_size <= 1:
            return self._predict_single(X, fallback_label, verbose, checkpoint_path, checkpoint_every)
        return self._predict_grouped(X, fallback_label, verbose, checkpoint_path, checkpoint_every)

    def _predict_single(self, X, fallback_label, verbose, checkpoint_path, checkpoint_every):
        n_total = len(X)
        y_pred = self._load_checkpoint(checkpoint_path, n_total, verbose) if checkpoint_path else None
        if y_pred is None:
            y_pred = [None] * n_total
        chunks_since_save = 0

        for start in range(0, n_total, self.batch_size):
            idxs = list(range(start, min(start + self.batch_size, n_total)))
            if all(y_pred[i] is not None for i in idxs):
                continue
            chunk = [X[i] for i in idxs]

            if verbose:
                print(f"[info] processing items {idxs[0]}-{idxs[-1]} of {n_total} (calls so far: {self.calls_made})")

            try:
                self.calls_made += len(chunk)
                results = self.chain.batch(
                    [{"tweet": t} for t in chunk],
                    config={"max_concurrency": self.max_concurrency},
                    return_exceptions=True,
                )
            except Exception as e:
                print(f"[warn] batch call failed entirely: {e}. Falling back to per-item retries.")
                results = [Exception(str(e))] * len(chunk)

            for i, res in zip(idxs, results):
                if isinstance(res, Exception):
                    y_pred[i] = self._invoke_with_retry(X[i], fallback_label=fallback_label, index=i)
                else:
                    y_pred[i] = res.label_id

            chunks_since_save += 1
            if checkpoint_path and chunks_since_save >= checkpoint_every:
                self._save_checkpoint(checkpoint_path, y_pred, n_total)
                chunks_since_save = 0

        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, y_pred, n_total)
        self._print_summary(verbose)
        return np.array(y_pred)

    def _predict_grouped(self, X, fallback_label, verbose, checkpoint_path, checkpoint_every):
        n_total = len(X)
        y_pred = self._load_checkpoint(checkpoint_path, n_total, verbose) if checkpoint_path else None
        if y_pred is None:
            y_pred = [None] * n_total

        groups = [(i, X[i:i + self.group_size]) for i in range(0, n_total, self.group_size)]
        chunks_since_save = 0

        for start in range(0, len(groups), self.batch_size):
            chunk_groups = groups[start:start + self.batch_size]

            # skip groups that are fully done already (checkpoint resume)
            pending = [(pos, g) for pos, g in chunk_groups if any(y_pred[pos + i] is None for i in range(len(g)))]
            if not pending:
                continue

            if verbose:
                print(
                    f"[info] processing groups at positions "
                    f"{pending[0][0]}-{pending[-1][0] + len(pending[-1][1]) - 1} of {n_total} "
                    f"(calls so far: {self.calls_made})"
                )

            group_inputs = [
                {"numbered_tweets": "\n".join(f'{i}: "{t}"' for i, t in enumerate(g)), "n_tweets": len(g)}
                for pos, g in pending
            ]

            try:
                self.calls_made += len(pending)
                results = self.group_chain.batch(
                    group_inputs,
                    config={"max_concurrency": self.max_concurrency},
                    return_exceptions=True,
                )
            except Exception as e:
                print(f"[warn] group batch call failed entirely: {e}. Falling back to per-group retries.")
                results = [Exception(str(e))] * len(pending)

            for (pos, g), res in zip(pending, results):
                if isinstance(res, Exception):
                    labels = self._invoke_group_with_retry(g, fallback_label=fallback_label, start_index=pos)
                else:
                    by_index = {r.index: r.label_id for r in res.results}
                    if set(by_index.keys()) == set(range(len(g))):
                        labels = [by_index[i] for i in range(len(g))]
                    else:
                        labels = self._invoke_group_with_retry(g, fallback_label=fallback_label, start_index=pos)
                for i, label in enumerate(labels):
                    y_pred[pos + i] = label

            chunks_since_save += 1
            if checkpoint_path and chunks_since_save >= checkpoint_every:
                self._save_checkpoint(checkpoint_path, y_pred, n_total)
                chunks_since_save = 0

        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, y_pred, n_total)
        self._print_summary(verbose)
        return np.array(y_pred)

    def _print_summary(self, verbose):
        if verbose:
            print(f"\n[summary] total API calls made: {self.calls_made}")
            if self.failed_indices:
                print(
                    f"[summary] {len(self.failed_indices)} tweet(s) could not be classified "
                    f"and were force-assigned a fallback label. Review manually: "
                    f"{self.failed_indices}"
                )


if __name__ == "__main__":
    # check your budget before running the full dataset, e.g.:
    estimate_calls(n_tweets=3718, group_size=10, daily_limit=500)

    sample_tweets = [
        "!!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house.",
        "!!!!! RT @mleew17: boy dats cold...tyga dwn bad for cuffin dat hoe in the 1st place!!",
        "Just had a great cup of coffee this morning, love this weather",
    ]

    model = LlmModel(batch_size=5, max_concurrency=5, group_size=10)
    preds = model.predict(sample_tweets, checkpoint_path="checkpoints/llm_predictions.json")
    print("Predictions:", preds)