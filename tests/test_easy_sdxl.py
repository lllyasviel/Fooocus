from local_markup.easy_sdxl import build_easy_sdxl_outputs, build_easy_sdxl_plan
from local_markup.style_explainer import describe_style_markdown, style_recommendations_for_goal


def test_easy_sdxl_edit_plan_keeps_sdxl_concepts():
    plan = build_easy_sdxl_plan("Edit This Image", "Glasses", "change the suit to a navy polo shirt")
    assert plan.recommended_tab == "Inpaint or Outpaint"
    assert "glasses" in plan.detection_prompt
    assert "shirt" in plan.detection_prompt
    assert "preserve facial identity" in plan.inpaint_prompt
    assert "CFG" in plan.checklist or plan.cfg == "4"
    assert "Inpaint" in plan.checklist


def test_easy_sdxl_outputs_order():
    positive, inpaint, detection, negative, settings, checklist = build_easy_sdxl_outputs(
        "Use Image as Reference",
        "Face",
        "professional headshot with natural light",
    )
    assert "professional headshot" in positive
    assert detection == "face"
    assert "Recommended tab" in settings
    assert "Image Prompt" in checklist
    assert isinstance(negative, str)


def test_headshot_instruction_produces_exact_edit_guidance():
    plan = build_easy_sdxl_plan(
        "Edit This Image",
        "I will draw the mask",
        "Remove glasses and change suit and tie to a clean navy blue polo shirt",
    )
    assert "without glasses" in plan.inpaint_prompt
    assert "navy blue polo shirt" in plan.inpaint_prompt
    assert "glasses" in plan.detection_prompt
    assert "shirt" in plan.detection_prompt
    assert "Do not mask the whole face" in plan.mask_tip


def test_remove_jacket_instruction_has_jacket_target():
    plan = build_easy_sdxl_plan(
        "Edit This Image",
        "Shirt / Clothes",
        "Remove only the jacket and replace with a clean professional shirt",
    )
    assert "shirt" in plan.detection_prompt
    assert "same person" in plan.inpaint_prompt
    assert "professional realistic headshot" in plan.inpaint_prompt


def test_style_explainer_returns_expectation_and_preview():
    markdown = describe_style_markdown("Fooocus Photograph", "professional realistic headshot")
    assert "Expectation" in markdown
    assert "Positive style effect" in markdown


def test_style_recommendations_for_headshot():
    recs = style_recommendations_for_goal("professional headshot")
    assert "Fooocus Photograph" in recs
    assert "Random Style" in recs
