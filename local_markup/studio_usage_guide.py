from __future__ import annotations


STUDIO_USAGE_GUIDE = """## Fastest way to use this Studio

1. Start Fooocus with `RUN_FOOOCUS_LOCAL.bat`.
2. Open Fooocus at `http://127.0.0.1:7865`.
3. Start AI Studio with `RUN_AI_STUDIO.bat`.
4. Describe one image you want.
5. Upload only the references that matter.
6. Click **Plan Best Fooocus Workflow**.
7. Copy the prompt, negative prompt, and selected Fooocus area into Fooocus.
8. Generate one image first.
9. Review it before making more versions.

Keep it simple: one goal, one first shot, one review cycle.
"""


NO_FRICTION_RULES = """## No-friction rules

- Do not guess which Fooocus tab to use; follow **Selected Fooocus workflow**.
- Do not copy every shot at once; start with the first prompt.
- Do not upload extra references; extra images create confusion.
- Do not change model/runtime settings while testing a prompt.
- If the first result is close, refine the prompt instead of starting over.
"""


def studio_usage_markdown() -> str:
    return f"{STUDIO_USAGE_GUIDE}\n\n{NO_FRICTION_RULES}"
