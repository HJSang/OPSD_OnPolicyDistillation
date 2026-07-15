#!/usr/bin/env python3
"""Score saved LongMemEval predictions with the official LLM-judge protocol."""

import argparse
import json
import os
from collections import defaultdict


DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
DEFAULT_REVISION = "1605565b47bb9346c5515c34102e054115b4f98b"

STANDARD_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only."
)
TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also answer "
    "yes. If the response only contains a subset of the information required by "
    "the answer, answer no. In addition, do not penalize off-by-one errors for the "
    "number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when "
    "the answer is 18), the model's response is still correct. \n\nQuestion: "
    "{}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
    "correct? Answer yes or no only."
)
UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with an "
    "updated answer, the response should be considered as correct as long as the "
    "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: "
    "{}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no "
    "only."
)
PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, and "
    "a response from a model. Please answer yes if the response satisfies the "
    "desired response. Otherwise, answer no. The model does not need to reflect "
    "all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: "
    "{}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)
ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information is "
    "not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the "
    "model correctly identify the question as unanswerable? Answer yes or no only."
)


def get_anscheck_prompt(question_type, question, answer, response, abstention=False):
    """Exact templates from LongMemEval evaluate_qa.py at commit 9e0b455."""
    if abstention:
        template = ABSTENTION_TEMPLATE
    elif question_type in {
        "single-session-user", "single-session-assistant", "multi-session"
    }:
        template = STANDARD_TEMPLATE
    elif question_type == "temporal-reasoning":
        template = TEMPORAL_TEMPLATE
    elif question_type == "knowledge-update":
        template = UPDATE_TEMPLATE
    elif question_type == "single-session-preference":
        template = PREFERENCE_TEMPLATE
    else:
        raise ValueError(f"unsupported LongMemEval question type: {question_type}")
    return template.format(question, answer, response)


def is_abstention(question_id):
    return "_abs" in question_id


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_prediction_key(records, requested):
    if requested:
        missing = [i for i, row in enumerate(records) if requested not in row]
        if missing:
            raise KeyError(f"{requested!r} is absent from record {missing[0]}")
        return requested
    candidates = set()
    for row in records:
        candidates.update(k for k in row if k == "pred" or k.endswith("_pred"))
    if len(candidates) != 1:
        raise ValueError(
            "could not infer one prediction field; pass --prediction-key "
            f"(candidates: {sorted(candidates)})"
        )
    return candidates.pop()


def attach_references(records, references):
    by_id = {row["question_id"]: row for row in references}
    by_qa = defaultdict(list)
    for row in references:
        by_qa[(row.get("question", ""), str(row.get("answer", "")))].append(row)

    attached = []
    for i, record in enumerate(records):
        qid = record.get("question_id")
        if not qid and isinstance(record.get("sample"), int):
            source = record["sample"]
            if 0 <= source < len(references):
                qid = references[source]["question_id"]
        if not qid:
            matches = by_qa[(record.get("question", ""), str(record.get("gold", "")))]
            if len(matches) != 1:
                raise ValueError(
                    f"record {i} has no question_id and matched {len(matches)} "
                    "reference rows"
                )
            qid = matches[0]["question_id"]
        if qid not in by_id:
            raise KeyError(f"record {i} references unknown question_id {qid!r}")
        attached.append((record, by_id[qid]))
    return attached


def render_chat_prompts(model, revision, prompts):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def score_with_vllm(args, prompts):
    from vllm import LLM, SamplingParams

    chat_prompts = render_chat_prompts(args.model, args.revision, prompts)
    llm = LLM(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=None if args.quantization == "none" else args.quantization,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
    )
    params = SamplingParams(temperature=0, max_tokens=10)
    outputs = llm.generate(chat_prompts, params)
    return [output.outputs[0].text.strip() for output in outputs]


def build_job(job, references):
    payload = load_json(job["input"])
    records = payload["records"] if isinstance(payload, dict) else payload
    prediction_key = select_prediction_key(records, job.get("prediction_key"))
    attached = attach_references(records, references)
    prompts = [
        get_anscheck_prompt(
            reference["question_type"],
            reference["question"],
            str(reference["answer"]),
            record[prediction_key],
            abstention=is_abstention(reference["question_id"]),
        )
        for record, reference in attached
    ]
    return {
        **job,
        "payload": payload,
        "prediction_key": prediction_key,
        "attached": attached,
        "prompts": prompts,
    }


def write_job(args, job, responses):
    by_category = defaultdict(list)
    judged_records = []
    for (record, reference), response in zip(job["attached"], responses):
        label = "yes" in response.lower()
        category = (
            "abstention"
            if is_abstention(reference["question_id"])
            else reference["question_type"]
        )
        row = dict(record)
        row["question_id"] = reference["question_id"]
        row["official_judge"] = {
            "model": args.model,
            "revision": args.revision,
            "response": response,
            "label": label,
        }
        judged_records.append(row)
        by_category[category].append(label)

    all_labels = [label for labels in by_category.values() for label in labels]
    metrics = {
        "overall": sum(all_labels) / len(all_labels),
        "by_category": {
            category: {"accuracy": sum(labels) / len(labels), "count": len(labels)}
            for category, labels in sorted(by_category.items())
        },
    }
    output = job.get("output")
    if not output:
        stem = os.path.splitext(job["input"])[0]
        output = f"{stem}_{job['prediction_key']}_official_judge.json"
    result = dict(job["payload"]) if isinstance(job["payload"], dict) else {}
    result["official_judge_config"] = {
        "protocol_source": "LongMemEval@9e0b455f4ef0e2ab8f2e582289761153549043fc",
        "model": args.model,
        "revision": args.revision,
        "temperature": 0,
        "max_tokens": 10,
        "tensor_parallel_size": args.tensor_parallel_size,
        "quantization": args.quantization,
        "prediction_key": job["prediction_key"],
    }
    result["official_metrics"] = metrics
    result["records"] = judged_records
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[judge] {job['prediction_key']} metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"[judge] wrote {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--prediction-key", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--jobs-file", default=None,
                        help="JSON list of input/prediction_key/output objects; "
                             "all jobs share one judge model load")
    parser.add_argument("--job", action="append", default=[],
                        metavar="INPUT:PREDICTION_KEY",
                        help="repeat for batched multi-file scoring with one "
                             "judge model load")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--quantization", default="fp8", choices=["fp8", "none"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.jobs_file:
        if args.input or args.prediction_key or args.output or args.job:
            parser.error("--jobs-file cannot be combined with single-job options")
        job_specs = load_json(args.jobs_file)
        if not isinstance(job_specs, list) or not job_specs:
            parser.error("--jobs-file must contain a non-empty JSON list")
    elif args.job:
        if args.input or args.prediction_key or args.output:
            parser.error("--job cannot be combined with single-job options")
        job_specs = []
        for value in args.job:
            if ":" not in value:
                parser.error("--job must use INPUT:PREDICTION_KEY syntax")
            path, prediction_key = value.rsplit(":", 1)
            job_specs.append({
                "input": path,
                "prediction_key": prediction_key,
            })
    else:
        if not args.input:
            parser.error("--input is required unless --jobs-file is used")
        job_specs = [{
            "input": args.input,
            "prediction_key": args.prediction_key,
            "output": args.output,
        }]
    references = load_json(args.data)
    jobs = [build_job(spec, references) for spec in job_specs]
    prompts = [prompt for job in jobs for prompt in job["prompts"]]
    for job in jobs:
        print(f"[judge] {len(job['prompts'])} predictions from {job['input']} "
              f"(field={job['prediction_key']})")
    print(f"[judge] model: {args.model}@{args.revision}")
    if args.dry_run:
        print(prompts[0] if prompts else "[judge] no records")
        return

    responses = score_with_vllm(args, prompts)
    offset = 0
    for job in jobs:
        count = len(job["prompts"])
        write_job(args, job, responses[offset:offset + count])
        offset += count


if __name__ == "__main__":
    main()
