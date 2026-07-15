# Context Soft-Token Compression

Research code for compressing conversational context into continuous soft tokens
and evaluating memory question answering. The main experiment compares raw text,
text summaries, visual/OCR compression, uniform soft-token pooling, and
role-aware soft-token pooling.

The repository is self-contained at the orchestration level: experiments are
plain Bash scripts, paths are detected from each script's location, missing
public benchmark data is downloaded automatically, and model aliases resolve to
public Hugging Face repositories.

See [the experiment guide](experiments/vtc_memory_validation/README.md) for setup
and commands.
