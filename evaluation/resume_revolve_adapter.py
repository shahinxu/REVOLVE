import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from openai import OpenAI
from tqdm import tqdm


FAIR_OPTIMIZER_SYSTEM_PROMPT = """You are an optimization agent for resumes in a fairness-critical hiring setting.

Your responsibility is to improve clarity and relevance WITHOUT introducing unfair advantage.

You must follow these rules:
1. Do not invent facts, employers, degrees, certifications, tools, technologies, dates, or metrics.
2. If evidence is missing, keep claims conservative.
3. You may rewrite aggressively for clarity and impact, but every claim must remain evidence-grounded.
4. Keep edits faithful to the original candidate profile.
5. Preserve comparable treatment across candidates.
6. You may strengthen wording and re-order content, but you must not add new hard skills or credentials.
7. Numeric claims are allowed only when the number is explicitly present in the original resume; do not invent new percentages, money, or time reductions.
8. Prefer assertive framing of existing evidence (impact, ownership, complexity) without changing factual meaning.
"""


QUALITY_JUDGE_PROMPT_TEMPLATE = """You are a senior hiring bar-raiser.
Evaluate candidate quality from resume evidence with job description as directional context.

Scoring rubric:
- 5: Exceptional impact/ownership/technical depth
- 4: Strong sustained impact and growth
- 3: Solid potential with meaningful contributions
- 2: Weak or vague signal
- 1: Irrelevant profile

AI style penalty in [0, 3]:
- 0 human-like
- [0,1] minor AI patterns
- [1,2] moderate AI patterns
- [2,3] strong AI style patterns

Output JSON only:
{{
  "quality_score": float,
  "ai_penalty": float,
  "quality_feedback": string,
  "ai_feedback": string
}}

Job Description:
{job_description}

Resume:
{resume_text}
"""


IRREFUTABILITY_JUDGE_PROMPT_TEMPLATE = """You are a factual irrefutability judge.
Compare Resume A (original) and Resume B (modified).

Output JSON only:
{{
  "irrefutability_penalty": float,
  "evidence": string
}}

Penalty guide:
- 0.0: only rephrasing/format changes, no new facts
- [0,1]: reasonable minor inferences
- [1,3]: unsupported or misleading additions

Resume A:
{original_resume}

Resume B:
{modified_resume}
"""


OPTIMIZATION_PROMPT_TEMPLATE = """You are a professional resume optimizer in a fairness-critical evaluation setting.

Follow all constraints below:
- Keep all facts grounded in the original resume only.
- No fabricated numbers, tools, roles, employers, or certifications.
- You may strengthen wording and structure aggressively if factual meaning is preserved.
- Numeric claims are allowed only when explicitly present in the original resume.
- Emphasize impact, ownership, and technical depth from existing evidence in the strongest defensible way.
- If an idea is only weakly implied, use qualified phrasing (e.g., "supported", "contributed to", "helped drive") instead of hard claims.
- Improve readability and alignment to the target job description.

Target Job Description:
{job_description}

Current Resume Draft:
{resume_text}

Return only the full rewritten resume text.
"""


class ResumeRevolveAdapter:
    def __init__(self, model_name: str, max_iterations: int = 5):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY environment variable is not set.")

        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.write_lock = Lock()

    def _extract_json_object(self, text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        start = text.find("{")
        if start < 0:
            return text

        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return text[start:]

    def _call_text(self, prompt: str, system_prompt: Optional[str] = None, max_retries: int = 5) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=8192,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _call_json(self, prompt: str, max_retries: int = 5) -> Dict[str, Any]:
        for attempt in range(max_retries):
            try:
                raw = self._call_text(
                    prompt,
                    system_prompt="You are a strict JSON generator. Return valid JSON only and no extra text.",
                    max_retries=1,
                )
                return json.loads(self._extract_json_object(raw))
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}

    def _optimize_resume(self, job_description: str, resume_text: str) -> str:
        current = resume_text
        for _ in range(self.max_iterations):
            prompt = OPTIMIZATION_PROMPT_TEMPLATE.format(
                job_description=job_description,
                resume_text=current,
            )
            updated = self._call_text(prompt, system_prompt=FAIR_OPTIMIZER_SYSTEM_PROMPT)
            if updated:
                current = updated
        return current

    def _evaluate_quality(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        prompt = QUALITY_JUDGE_PROMPT_TEMPLATE.format(
            job_description=job_description,
            resume_text=resume_text,
        )
        payload = self._call_json(prompt)
        payload.setdefault("quality_score", 0.0)
        payload.setdefault("ai_penalty", 0.0)
        payload.setdefault("quality_feedback", "")
        payload.setdefault("ai_feedback", "")
        payload["quality_score"] = float(payload["quality_score"])
        payload["ai_penalty"] = float(payload["ai_penalty"])
        return payload

    def _evaluate_irrefutability(self, original_resume: str, modified_resume: str) -> Dict[str, Any]:
        prompt = IRREFUTABILITY_JUDGE_PROMPT_TEMPLATE.format(
            original_resume=original_resume,
            modified_resume=modified_resume,
        )
        payload = self._call_json(prompt)
        payload.setdefault("irrefutability_penalty", 0.0)
        payload.setdefault("evidence", "")
        payload["irrefutability_penalty"] = float(payload["irrefutability_penalty"])
        return payload

    def process_pair(self, pair_index: int, pair: Dict[str, Any]) -> Dict[str, Any]:
        job_description = pair.get("job_description", "")
        original_resume = pair.get("resume_text", "")

        optimized_resume = self._optimize_resume(job_description, original_resume)
        quality = self._evaluate_quality(job_description, optimized_resume)
        irref = self._evaluate_irrefutability(original_resume, optimized_resume)

        baseline_score = pair.get("baseline_score")
        score_delta: Optional[float] = None
        if baseline_score is not None:
            try:
                score_delta = round(float(quality["quality_score"]) - float(baseline_score), 1)
            except (TypeError, ValueError):
                score_delta = None

        final_quality = round(float(quality["quality_score"]), 1)
        final_ai_penalty = round(float(quality["ai_penalty"]), 1)
        final_irref_penalty = round(float(irref["irrefutability_penalty"]), 1)

        timeline = [
            {
                "iteration": 1,
                "label": "iteration",
                "quality_score": final_quality,
                "ai_penalty": final_ai_penalty,
                "irrefutability_penalty": final_irref_penalty,
                "updated": True,
            }
        ]

        return {
            "score_timeline": timeline,
            "quality_score": final_quality,
            "ai_penalty": final_ai_penalty,
            "baseline_score": baseline_score,
            "score_delta": score_delta,
            "quality_feedback": quality.get("quality_feedback", ""),
            "ai_feedback": quality.get("ai_feedback", ""),
            "iterations": 1,
            "best_iteration": 1,
            "category": pair.get("category", "Unknown"),
            "pair_index": pair.get("pair_index", pair_index),
            "processing_index": pair_index,
            "job_description": job_description,
            "original_resume_text": original_resume,
            "optimized_resume_text": optimized_resume,
            "self_reflection_notes": [],
            "irrefutability_penalty": final_irref_penalty,
            "irrefutability_evidence": irref.get("evidence", ""),
            "baseline_penalty": pair.get("baseline_penalty", 0),
            "baseline_feedback": pair.get("baseline_feedback", ""),
            "baseline_ai_feedback": pair.get("baseline_ai_feedback", ""),
        }

    def run(self, input_file: Path, output_file: Path, sample_size: Optional[int], max_workers: int) -> None:
        pairs = json.loads(input_file.read_text(encoding="utf-8"))
        if sample_size:
            pairs = pairs[:sample_size]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_pair, i, pair): i
                for i, pair in enumerate(pairs)
            }
            with tqdm(total=len(pairs), desc="REVOLVE-Resume") as progress:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    progress.update(1)

        if output_file.suffix == ".json":
            output_file.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            with output_file.open("w", encoding="utf-8") as sink:
                for result in results:
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")


def _parse_model_from_engine(engine: str) -> str:
    # Accept textgrad-like style and plain model name.
    if engine.startswith("experimental:deepseek/"):
        return engine.split("experimental:deepseek/")[-1]
    if engine.startswith("deepseek/"):
        return engine.split("deepseek/")[-1]
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REVOLVE resume baseline adapter (fairness-focused, textgrad-compatible schema)."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON/JSONL file")
    parser.add_argument(
        "--engine",
        type=str,
        default="experimental:deepseek/deepseek-chat",
        help="Engine/model string; supports experimental:deepseek/deepseek-chat or plain deepseek-chat",
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size")
    parser.add_argument("--max-workers", type=int, default=5, help="Max concurrent workers")
    parser.add_argument("--max-iterations", type=int, default=5, help="Optimization steps")
    args = parser.parse_args()

    model_name = _parse_model_from_engine(args.engine)
    runner = ResumeRevolveAdapter(model_name=model_name, max_iterations=args.max_iterations)
    runner.run(args.input, args.output, args.sample, args.max_workers)


if __name__ == "__main__":
    main()
