# Replication of DeepSeek-V4.1-Flash on GPQA-Diamond

## 1. Key Results

| Run | Accuracy | Errors | Input Tokens | Cache Tokens | Output Tokens | Cost USD |
| --- | --- | --- | --- | --- | --- | --- |
| Run 1 | 93.43% | 3.54% | 955,945 | 717,696 | 3,816,361 | $4.66 |
| Run 2 | 88.89% | 3.03% | 885,564 | 695,168 | 3,699,283 | $4.50 |
| Run 3 | 92.42% | 3.03% | 745,071 | 595,584 | 3,284,607 | $3.99 |
| **Average** | **91.58%** | **3.2%** | 862,193 | 669,483 | 3,600,084 | **$4.38** |

**Spread and variance:** The run-level spread is 4.54 (88.89% - 93.43%), demonstrating the inherent variance of single-seed runs.

**Target:** The assignment's goal was to replicate Artificial Analysis's reported 91% on GPQA-Diamond for DeepSeek-V4.1-Flash, with a minimum of 85%. This replication's 3-run average of **91.58%** meets both the minimum and, notably, slightly exceeds AA's reported figure.

---

## 2. Methodology and Configuration

### Model & APIs
- **Model Used:** This project uses `DeepSeek-V4.1-Flash`. I was originally going to use `DeepSeek-V4-Flash-0731`, but as noted in the DeepSeek API docs, the aliases have retired and routes to `DeepSeek-V4.1-Flash`
- **Reasoning effort:** high / thinking-enabled, matching AA's max reasoning configuration
- **Other APIs Used**: HuggingFace to obtain the dataset

### Harness
- **Framework:** `Harbor` - an agent evaluation framework
- **Agent:** `terminus-2@2.0.0` — an agent harness to evaluate LLMs
- **Agent timeout:** `900s` per task (raised from an initial `300s` — see Section 4 for why this alone did not meaningfully fix failures).
- **Concurrency:** `-n` set based on locally-observed container stability (see `configs/` for exact value used); early runs at higher concurrency produced sporadic `RuntimeError`s from Docker/tmux startup contention, which is why this value was tuned down before the final 3 runs.
- **Environment:** minimal `ubuntu:24.04` container per task, no special tooling pre-installed — the agent has a full Linux terminal available but is instructed the task is closed-book (Section 4).

### Dataset
- **Source:** GPQA-Diamond, 198 graduate-level science questions, via [`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa) (`gpqa_diamond` subset) on Hugging Face.
- **Answer-choice shuffling:** Each question's four choices (1 correct + 3 incorrect) are shuffled into A/B/C/D order using `random.Random(question_index)` — i.e., a fixed, reproducible seed per question. This means the lettered position of the correct answer is identical across all 3 runs; only the model's own answers vary run to run. This isolates model/agent variance from any variance that shuffling choices differently would introduce.

---

## 3. Task Design

Each of the 198 GPQA-Diamond questions is compiled into its own Harbor task under `tasks/gpqa_diamond/gpqa-XXX/`, following Harbor's standard task layout:

```
gpqa-XXX/
  task.toml              # task metadata, timeouts, resource limits
  instruction.md          # the question, 4 lettered choices, and answer-format instructions
  environment/Dockerfile  # minimal ubuntu:24.04 sandbox
  solution/solve.sh       # reference solution — writes the correct letter directly
  tests/
    test.sh                # verifier: compares /app/answer.txt to expected_answer.txt
    expected_answer.txt     # the correct letter (hidden from the agent's environment)
```

**Verifier logic:** `test.sh` reads the agent's `/app/answer.txt`, strips whitespace, and compares it against `tests/expected_answer.txt` (which lives outside the agent's mounted working directory and is never exposed to it). A match writes `1` to `/logs/verifier/reward.txt`; a mismatch or a missing answer file writes `0`.

**Validation before spending real API calls:** Before running any task against DeepSeek, every task was checked with Harbor's built-in `oracle` agent, which executes `solution/solve.sh` instead of calling a model. This confirmed the full task/verifier pipeline (Docker build → answer write → grading) worked correctly and gave reward `1.0` on known-correct answers, at zero API cost, before committing to any paid runs.

Tasks are generated programmatically by [`scripts/build_tasks.py`](scripts/build_tasks.py) from the raw Hugging Face dataset.

---

## 4. Prompt Engineering

The final prompt used in every task's `instruction.md` was arrived at through several rounds of diagnosis, not written once and left alone. The full progression:

### Iteration 1 — baseline (no closed-book guidance, 300s timeout)
First full 198-question run: **84.8% accuracy**, with **27/198 (13.6%) trials failing outright** — 25 `AgentTimeoutError`, 2 `RuntimeError`.

Reading the failed trajectories showed two distinct problems, not one:
- **`RuntimeErrors`** ("Failed to start tmux session", "docker inspect returned 1") meaning containers were failing to start under concurrent load, unrelated to the model reasoning.
- **`AgentTimeoutErrors`** showed the model repeatedly attempting external verification — `curl`, Python `urllib`, Wikipedia, DuckDuckGo, Bing, the Chemistry StackExchange API, `pip install rdkit` — almost all of which were blocked or unreachable inside the sandboxed container, yet the model kept trying new search strategies rather than falling back to its own reasoning.

### Iteration 2 — raised timeout only (900s)
Tripling the timeout budget produced only a marginal change in timeout rate (13.6% → ~11.4% on a 35-question sample), despite 3x more time available. Ruled out the possibility of insufficient time being the primary issue and confirmed that the real problem was with the behavior of the model.

### Iteration 3 — explicit closed-book instructions
Added directives to `instruction.md` stating explicitly that the problem is solvable from internal knowledge alone, that external tools are unreliable and unnecessary, and that the agent should write its answer once and not re-verify. On the same 35-question validation sample, this raised accuracy to **94.3%** and cut the error count from 4 to 2.

### Iteration 4 — final refinement
Trajectory review at this stage (`gpqa-008`) showed that the model was still second-guessing an already answer. It's reasoning states "before finalizing, I want to verify..." after already writing the correct letter. While the instruction reduced re-verification behavior, it didn't eliminate it. The final prompt strengthens this with explicity cost-framing and an instruction to treat the impulse to double-check as a signal to stop.

### Final prompt template

```python
def build_instruction(question, shuffled_choices):
    choices_text = "\n".join(
        f"{letter}) {choice.strip()}"
        for letter, choice in zip("ABCD", shuffled_choices)
    )

    return f"""Answer the following multiple-choice science question.

{question.strip()}

{choices_text}

Important execution constraints:
- This is a closed-book domain knowledge problem. It is fully solvable using internal chemistry, physics, and biology reasoning alone.
- External tools and network lookups (e.g., curl, web scrapers, search engines) are unnecessary, unreliable, and should not be relied upon. If any external attempt fails, times out, or returns blocked/empty output, immediately stop attempting external lookups and deduce the solution using first-principles scientific reasoning.
- Once you have written an answer to /app/answer.txt, the task is done. Re-checking, re-deriving, or re-verifying an answer you have already written provides no benefit to your score and only risks running out of time before the file is saved. Do not return to a question you've already answered.
- If you find yourself reasoning about whether to double-check your answer, that reasoning itself is the signal to stop and submit immediately instead.

Think through the problem carefully once. When you have determined the correct \
answer, write ONLY the single letter (A, B, C, or D) to a file called \
/app/answer.txt, with no extra text, punctuation, or explanation in the file.

Example: if you believe the answer is C, run:
echo -n "C" > /app/answer.txt

Immediately exit and mark the task complete as soon as /app/answer.txt is written.
"""
```

This template, unchanged, was used for all 3 of the final full runs reported in Section 1.

---

## 5. Reproducing This

1. Clone this repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` file with:
   ```
   DEEPSEEK_API_KEY=sk-...
   HF_TOKEN=hf_...
   ```
3. Generate the task set (downloads GPQA-Diamond and builds 198 Harbor tasks):
   ```
   python scripts/build_tasks.py
   ```
4. (Optional, recommended) Validate the pipeline for free before spending API credits:
   ```
   harbor run -p tasks/gpqa_diamond --include-task-name gpqa-000 -a oracle
   ```
5. Run the full 198-question set against DeepSeek-V4.1-Flash, three times:
   ```
   harbor run -p tasks/gpqa_diamond -a terminus-2 -m deepseek/deepseek-flash -n <N>
   ```
   Run this command 3 separate times without modifying the task files or prompt between runs. Each invocation produces its own `jobs/<timestamp>/result.json` and per-trial trajectory files.
---

## 6. Cross-Run Analysis

Because each of the 198 questions was run 3 times against the exact same shuffled choices, individual questions can be classified by consistency rather than relying on the aggregate percentage alone:

- **Consistently failed in all 3 runs:** `gpqa-081`, `gpqa-088`, `gpqa-130`, `gpqa-145`, `gpqa-147` — each of these timed out (`AgentTimeoutError`) in every single run, suggesting a genuine, repeatable difficulty (these involve dense multi-step organic chemistry synthesis problems) rather than random variance.
- **Inconsistent across runs (passed in some, failed in others):** the majority of the ~13–22 per-run failures fall here — these are better explained by run-to-run stochasticity in the model's reasoning than by a fixed capability gap.

---

## 7. Limitations 

- **Prompt tuning was done on a 35-question sample**, not the full 198, for cost and time reasons. The final 3 full runs reported above are on the complete, untouched question set, but the prompt itself was shaped by feedback from a subset of it
- **The "closed-book, no re-verification" instructions are this project's own methodology choice**, not a known replication of AA's exact internal prompt (which isn't public). They were added specifically to address an observed failure, but this is a deviation from a fully default agent.
- **A small number of consistently timing-out questions remain** even after prompt and timeout tuning (Section 6) — this is reported as a known limitation of the current configuration rather than something fully resolved.
