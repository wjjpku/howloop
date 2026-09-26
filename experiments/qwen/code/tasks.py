from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
import re
from typing import Callable, Iterable


TASK_NAMES = ("successor", "weekday", "doubling", "fibonacci", "collatz")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
NUM_LOOPS = 4


@dataclass(frozen=True)
class Example:
    task: str
    start: object
    steps: int
    prompt: str
    answer: str
    trace: tuple[object, ...]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def collatz_step(value: int) -> int:
    return value // 2 if value % 2 == 0 else 3 * value + 1


def fibonacci_pair(index: int) -> tuple[int, int]:
    a, b = 0, 1
    for _ in range(index):
        a, b = b, a + b
    return a, b


def iterate(value, transition: Callable, steps: int) -> tuple[object, tuple[object, ...]]:
    trace: list[object] = []
    for _ in range(steps):
        value = transition(value)
        trace.append(value)
    return value, tuple(trace)


def direct_answer_suffix() -> str:
    return "Answer directly with only the answer. Answer:"


def make_example(
    task: str,
    rng: random.Random,
    *,
    start_override: object | None = None,
) -> Example:
    steps = NUM_LOOPS
    step_text = f"{steps:02d}"
    suffix = direct_answer_suffix()
    if task == "successor":
        start = rng.randint(0, 99) if start_override is None else int(start_override)
        answer, trace = iterate(start, lambda value: value + 1, steps)
        prompt = (
            f"Starting with {start}, apply the integer successor operation exactly {step_text} times. "
            f"What is the final value? {suffix}"
        )
    elif task == "weekday":
        if start_override is None:
            start = WEEKDAYS[rng.randrange(len(WEEKDAYS))]
        else:
            start = str(start_override)
        answer, trace = iterate(
            start,
            lambda value: WEEKDAYS[(WEEKDAYS.index(value) + 1) % len(WEEKDAYS)],
            steps,
        )
        prompt = (
            f"Starting with {start}, move forward exactly {step_text} days. "
            f"What is the final weekday? {suffix}"
        )
    elif task == "doubling":
        start = rng.randint(1, 9) if start_override is None else int(start_override)
        answer, trace = iterate(start, lambda value: 2 * value, steps)
        prompt = (
            f"Starting with {start}, double the value exactly {step_text} times. "
            f"What is the final value? {suffix}"
        )
    elif task == "fibonacci":
        pair_index = rng.randint(0, 8) if start_override is None else int(start_override)
        start = fibonacci_pair(pair_index)
        final_pair, pair_trace = iterate(start, lambda pair: (pair[1], pair[0] + pair[1]), steps)
        answer = final_pair[1]
        trace = tuple(pair[1] for pair in pair_trace)
        prompt = (
            f"Starting with the Fibonacci pair {start[0]}, {start[1]}, advance the recurrence exactly "
            f"{step_text} times. What is the final second value? {suffix}"
        )
    elif task == "collatz":
        start = rng.randint(2, 99) if start_override is None else int(start_override)
        answer, trace = iterate(start, collatz_step, steps)
        prompt = (
            f"Starting with {start}, apply the standard Collatz step exactly {step_text} times. "
            f"What is the final value? {suffix}"
        )
    else:
        raise ValueError(f"unknown task {task!r}")
    example = Example(
        task=task,
        start=start,
        steps=steps,
        prompt=prompt,
        answer=str(answer),
        trace=trace,
    )
    assert_example(example)
    return example


def assert_example(example: Example) -> None:
    if example.steps != NUM_LOOPS or len(example.trace) != NUM_LOOPS:
        raise RuntimeError("every task example must execute exactly four transitions")
    if str(example.trace[-1]) != example.answer:
        raise RuntimeError("serialized target is not x4")


def build_bank(*, count_per_task: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples: list[Example] = []
    for task in TASK_NAMES:
        for _ in range(count_per_task):
            examples.append(make_example(task, rng))
    return examples


def build_enumerated_bank() -> list[Example]:
    rng = random.Random(0)
    starts: dict[str, Iterable[object]] = {
        "successor": range(0, 100),
        "weekday": WEEKDAYS,
        "doubling": range(1, 10),
        "fibonacci": range(0, 9),
        "collatz": range(2, 100),
    }
    return [
        make_example(task, rng, start_override=start)
        for task in TASK_NAMES
        for start in starts[task]
    ]


def examples_sha256(examples: Iterable[Example]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.canonical_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_first_answer(text: str, task: str):
    if task == "weekday":
        match = re.search("|".join(WEEKDAYS), text, flags=re.IGNORECASE)
        return match.group(0).lower() if match else None
    match = re.search(r"(?<!\w)-?\d[\d,]*(?!\w)", text)
    return int(match.group(0).replace(",", "")) if match else None
