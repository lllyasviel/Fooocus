from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from local_markup.ai_studio_agent_v2 import AgentPlan, build_agent_plan
from local_markup.engine_hardware_profiles import HardwareProfile, profile_summary, select_hardware_profile
from local_markup.engine_queue_contract import EngineJobKind
from local_markup.local_fooocus_adapter import LocalDryRunFooocusAdapter
from local_markup.studio_adapter_contract import AdapterResult, ImageStudioJob, ReferenceImage
from local_markup.studio_adapter_mappings import build_face_reference_job, build_image_prompt_job, build_inpaint_job
from local_markup.studio_generation_history import add_adapter_result_to_history
from local_markup.studio_history import StudioHistoryStore


REFERENCE_PREFIX = "studio_reference"


@dataclass(frozen=True)
class StudioWorkflowRun:
    plan: AgentPlan
    job: ImageStudioJob
    adapter_result: AdapterResult
    hardware_profile: HardwareProfile
    history: StudioHistoryStore
    mapping_notes: List[str]

    def adapter_markdown(self) -> str:
        reference_lines = [f"- {item.role}: `{item.path or item.name}`" for item in self.job.references]
        if not reference_lines:
            reference_lines = ["- No references needed for this first shot."]

        setup_steps = [
            f"Open Fooocus area: **{self.plan.fooocus_tab}**.",
            "Paste the first prompt into the main prompt box.",
            "Paste the negative prompt into the negative prompt box.",
            "Upload the listed reference images in the same order.",
            "Generate one image first, then review before continuing.",
        ]
        handoff_steps = "\n".join([f"{index + 1}. {step}" for index, step in enumerate(setup_steps)])
        notes = "\n".join([f"- {note}" for note in self.mapping_notes])
        return (
            "## Ready for Fooocus\n\n"
            "This is a safe preview. Nothing was generated yet.\n\n"
            f"**Workflow:** {self.plan.tool}\n\n"
            f"**Fooocus area:** {self.plan.fooocus_tab}\n\n"
            f"**Hardware recommendation:** {profile_summary(self.hardware_profile)}\n\n"
            f"### References to upload\n{chr(10).join(reference_lines)}\n\n"
            f"### Simple setup steps\n{handoff_steps}\n\n"
            f"### Why this setup\n{notes or '- Standard text-to-image setup.'}"
        )

    def history_markdown(self) -> str:
        rows = []
        for item in self.history.latest(limit=5):
            rows.append(
                f"- `{item.workflow}` plan saved as `{item.item_id}` with {item.metadata.get('reference_count', '0')} reference(s). Status: {item.metadata.get('adapter_status', 'unknown')}."
            )
        return "## Local Session History\n\n" + ("\n".join(rows) if rows else "No session history yet.")


def reference_names_from_count(image_count: int) -> List[str]:
    return [f"{REFERENCE_PREFIX}_{index + 1}" for index in range(max(0, image_count))]


def references_from_count(image_count: int, role: str = "reference") -> List[ReferenceImage]:
    return [ReferenceImage(name=name, path=name, role=role) for name in reference_names_from_count(image_count)]


def kind_for_feature(feature_key: str) -> EngineJobKind:
    if feature_key in {"inpaint", "auto_mask_sam", "auto_mask_u2net"}:
        return EngineJobKind.INPAINT
    if feature_key in {"upscale", "variation"}:
        return EngineJobKind.UPSCALE
    if feature_key == "enhance":
        return EngineJobKind.ENHANCE
    if feature_key in {"image_prompt", "face_reference", "pyracanny", "cpds", "describe"}:
        return EngineJobKind.IMAGE_PROMPT
    return EngineJobKind.TEXT_TO_IMAGE


def build_job_from_plan(plan: AgentPlan, image_count: int) -> tuple[ImageStudioJob, List[str]]:
    feature_key = plan.feature.key
    refs = reference_names_from_count(image_count)

    if feature_key == "face_reference" and refs:
        mapping = build_face_reference_job(plan.user_goal, plan.primary_prompt, plan.negative_prompt, refs[0])
        return mapping.job, [note.note for note in mapping.notes]

    if feature_key in {"image_prompt", "pyracanny", "cpds", "variation", "describe"} and refs:
        mapping = build_image_prompt_job(plan.user_goal, plan.primary_prompt, plan.negative_prompt, refs)
        metadata = dict(mapping.job.metadata)
        metadata["studio_feature"] = feature_key
        job = ImageStudioJob(
            goal=mapping.job.goal,
            prompt=mapping.job.prompt,
            negative_prompt=mapping.job.negative_prompt,
            kind=mapping.job.kind,
            references=mapping.job.references,
            width=mapping.job.width,
            height=mapping.job.height,
            seed=mapping.job.seed,
            metadata=metadata,
        )
        return job, [note.note for note in mapping.notes]

    if feature_key in {"inpaint", "auto_mask_sam", "auto_mask_u2net"}:
        source = refs[0] if refs else "source_image_required"
        mask = refs[1] if len(refs) > 1 else "mask_required"
        mapping = build_inpaint_job(plan.user_goal, plan.primary_prompt, plan.negative_prompt, source, mask)
        return mapping.job, [note.note for note in mapping.notes]

    references = references_from_count(image_count)
    job = ImageStudioJob(
        goal=plan.user_goal,
        prompt=plan.primary_prompt,
        negative_prompt=plan.negative_prompt,
        kind=kind_for_feature(feature_key),
        references=references,
        metadata={"fooocus_area": plan.fooocus_tab, "studio_feature": feature_key},
    )
    notes = [f"Use the `{feature_key}` Studio plan as a `{job.kind.value}` Fooocus job."]
    return job, notes


def run_studio_workflow(
    goal: str,
    image_count: int = 0,
    wants_identity: bool = True,
    wants_exact_edit: bool = False,
    wants_bundle: bool = False,
    vram_gb: int = 6,
    existing_history: Optional[StudioHistoryStore] = None,
) -> StudioWorkflowRun:
    plan = build_agent_plan(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)
    job, mapping_notes = build_job_from_plan(plan, image_count)
    hardware_profile = select_hardware_profile(vram_gb)
    adapter_result = LocalDryRunFooocusAdapter().submit(job)
    history = add_adapter_result_to_history(existing_history or StudioHistoryStore(), job, adapter_result)
    return StudioWorkflowRun(
        plan=plan,
        job=job,
        adapter_result=adapter_result,
        hardware_profile=hardware_profile,
        history=history,
        mapping_notes=mapping_notes,
    )


def build_studio_workflow_outputs(
    goal: str,
    image_1,
    image_2,
    image_3,
    wants_identity: bool,
    wants_exact_edit: bool,
    wants_bundle: bool,
    vram_gb: int = 6,
) -> tuple[str, str, str, str, str, str, str, str, str]:
    image_count = sum(x is not None for x in [image_1, image_2, image_3])
    run = run_studio_workflow(goal, image_count, wants_identity, wants_exact_edit, wants_bundle, vram_gb=vram_gb)
    shot_list = "\n\n".join([f"Shot {index + 1}: {prompt}" for index, prompt in enumerate(run.plan.shot_prompts)])
    return (
        run.plan.as_markdown(),
        run.plan.primary_prompt,
        run.plan.negative_prompt,
        run.plan.tool,
        run.plan.fooocus_tab,
        shot_list,
        run.plan.handoff_recipe,
        run.adapter_markdown(),
        run.history_markdown(),
    )
