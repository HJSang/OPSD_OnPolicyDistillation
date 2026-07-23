"""Prompt helpers for inserting continuous context into chat templates."""

SOFT_CONTEXT_MARKER = "<|soft_context_placeholder_7f3a9d|>"


def render_user_prompt(tokenizer, user_text):
    """Render one user turn followed by the model's assistant-generation prefix."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "The tokenizer has no chat_template; use prompt_format='plain' "
            "only for legacy checkpoints."
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def split_chat_prompt(tokenizer, user_text):
    """Render a chat prompt and split it around the soft-context placeholder."""
    if user_text.count(SOFT_CONTEXT_MARKER) != 1:
        raise ValueError("user_text must contain exactly one soft-context marker")
    rendered = render_user_prompt(tokenizer, user_text)
    if rendered.count(SOFT_CONTEXT_MARKER) != 1:
        raise ValueError("chat template changed or duplicated the soft-context marker")
    return rendered.split(SOFT_CONTEXT_MARKER, 1)


def resolve_prompt_format(requested, checkpoint_args=None):
    """Resolve ``auto`` while retaining compatibility with old checkpoints."""
    if requested != "auto":
        return requested
    if checkpoint_args is None:
        return "chat"
    return checkpoint_args.get("prompt_format", "plain")
