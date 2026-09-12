from .planner import build_edit_intent


def build_markup_ui_outputs(user_instruction):
    intent = build_edit_intent(user_instruction)
    plan = f"Action: {intent.action}\nMode: {intent.mode}\nTarget: {', '.join(intent.targets)}\nPrompt:\n{intent.prompt}"
    return intent.prompt, plan
