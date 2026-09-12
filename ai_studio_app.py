import gradio as gr

from local_markup.studio_workflow_controller import build_studio_workflow_outputs
from local_markup.fooocus_feature_playbook import build_feature_reasoning
from local_markup.fooocus_feature_catalog import list_features_markdown
from local_markup.style_explainer import describe_style_markdown, list_style_names, style_recommendations_for_goal
from local_markup.studio_usage_guide import studio_usage_markdown
from local_markup.studio_one_ui_note import FOOOCUS_ENGINE_URL, one_ui_note_markdown
from local_markup.studio_control_ui import (
    CONTROL_UI_CSS,
    engine_hidden_note,
    history_gallery_empty_note,
    launcher_controls_markdown,
    studio_hero_markdown,
    studio_status_panel_html,
)
from local_markup.studio_copy_controls import copy_controls_summary
from local_markup.studio_downloads import build_engine_handoff_text, write_history_download, write_prompt_pack


STYLE_NAMES = list_style_names()


def copyable_textbox(**kwargs):
    return gr.Textbox(show_copy_button=True, **kwargs)


def explain_style(style_name, goal):
    return describe_style_markdown(style_name, goal or "realistic personal photo")


def recommend_styles(goal):
    return style_recommendations_for_goal(goal or "realistic personal photo")


def explain_feature_stack(goal, image_1, image_2, image_3, wants_identity, wants_exact_edit, wants_bundle):
    image_count = sum(x is not None for x in [image_1, image_2, image_3])
    return build_feature_reasoning(goal, image_count, wants_identity, wants_exact_edit, wants_bundle)


def feature_catalog_markdown():
    return list_features_markdown()


def fooocus_iframe_html():
    return (
        f'<iframe id="fooocus_engine_iframe" src="{FOOOCUS_ENGINE_URL}" '
        'style="width:100%; height:78vh; border:1px solid #333; border-radius:12px;" '
        'title="Fooocus Engine"></iframe>'
    )


def studio_engine_bridge_script():
    return f"""
<script>
(function() {{
    if (window.__fooocusStudioEngineBridgeInstalled) return;
    window.__fooocusStudioEngineBridgeInstalled = true;

    const ENGINE_ORIGIN = new URL("{FOOOCUS_ENGINE_URL}").origin;
    let planReadySeen = false;
    let waitingForEngineAck = false;

    function byId(id) {{ return document.getElementById(id); }}
    function readTextbox(elemId) {{
        const root = byId(elemId);
        if (!root) return "";
        const field = root.querySelector("textarea") || root.querySelector("input");
        return field ? field.value : "";
    }}
    function markStep(stepId, done) {{
        const step = byId(stepId);
        if (step) step.classList.toggle("done", !!done);
    }}
    function showToast(message) {{
        const toast = byId("studio_toast");
        if (!toast) return;
        toast.textContent = message;
        toast.classList.add("show");
        window.clearTimeout(window.__fooocusStudioToastTimer);
        window.__fooocusStudioToastTimer = window.setTimeout(() => toast.classList.remove("show"), 5000);
    }}
    function setStatus(stage, title, message, percent, toastMessage) {{
        const statusTitle = byId("studio_status_title");
        const statusMessage = byId("studio_status_message");
        const fill = byId("studio_progress_fill");
        const dot = byId("studio_status_dot");
        if (statusTitle) statusTitle.textContent = title;
        if (statusMessage) statusMessage.textContent = message;
        if (fill) fill.style.width = `${{Math.max(0, Math.min(100, percent || 0))}}%`;
        if (dot) dot.classList.toggle("active", stage !== "idle");
        markStep("studio_step_plan", percent >= 35);
        markStep("studio_step_send", percent >= 65);
        markStep("studio_step_ack", percent >= 95);
        markStep("studio_step_generate", percent >= 100);
        if (toastMessage) showToast(toastMessage);
    }}
    function planHasOutput() {{ return readTextbox("studio_primary_prompt").trim().length > 0; }}
    function watchPlanReady() {{
        if (planReadySeen || !planHasOutput()) return;
        planReadySeen = true;
        setStatus("planned", "Plan ready", "Studio produced the prompt, negative prompt, workflow, and setup fields. Click Send to Engine next.", 35, "Plan ready. Send it to the engine.");
    }}
    function postStudioPayloadToEngine(payload) {{
        const iframe = byId("fooocus_engine_iframe");
        if (!iframe || !iframe.contentWindow) {{
            setStatus("missing-engine", "Engine iframe not ready", "Open the Hidden Fooocus engine panel or restart with START_AI_IMAGE_STUDIO.bat, then send again.", 35, "Engine iframe was not ready.");
            return false;
        }}
        waitingForEngineAck = true;
        setStatus("sending", "Sending to engine", "Studio is sending the prompt and negative prompt into the embedded Fooocus engine. Waiting for field-fill confirmation.", 65, "Sending fields to Fooocus.");
        iframe.contentWindow.postMessage(payload, ENGINE_ORIGIN);
        window.clearTimeout(window.__fooocusStudioAckTimer);
        window.__fooocusStudioAckTimer = window.setTimeout(function() {{
            if (!waitingForEngineAck) return;
            setStatus("sent-waiting", "Sent, waiting for confirmation", "The message was sent. If the hidden engine was still loading, open that panel and click Send to Engine again.", 70, "Sent. Waiting for the engine to confirm field fill.");
        }}, 2200);
        return true;
    }}
    function buildStudioPayloadFromFields() {{
        return {{
            type: "fooocus-studio-autofill",
            workflow: readTextbox("studio_selected_tool"),
            fooocus_area: readTextbox("studio_selected_area"),
            prompt: readTextbox("studio_primary_prompt"),
            negative_prompt: readTextbox("studio_negative_prompt"),
            setup_steps: readTextbox("studio_handoff_recipe"),
            next_shots: readTextbox("studio_shot_prompts")
        }};
    }}
    function sendStudioPlanToEngine() {{ return postStudioPayloadToEngine(buildStudioPayloadFromFields()); }}
    window.fooocusStudioSetStatus = setStatus;
    window.fooocusStudioSendToEngine = sendStudioPlanToEngine;
    window.fooocusStudioPostPayloadToEngine = postStudioPayloadToEngine;
    document.addEventListener("click", function(event) {{
        if (event.target.closest("#studio_build_plan_button")) {{
            planReadySeen = false;
            setStatus("planning", "Planning image workflow", "Studio is selecting the Fooocus workflow and preparing fields. This progress reflects planning only.", 12, "Building plan.");
        }}
    }}, true);
    window.addEventListener("message", function(event) {{
        if (event.origin !== ENGINE_ORIGIN) return;
        const payload = event.data || {{}};
        if (payload.type !== "fooocus-studio-autofill-result") return;
        waitingForEngineAck = false;
        const ok = !!payload.promptFilled && !!payload.negativeFilled;
        if (ok) {{
            setStatus("filled", "Engine fields filled", "Fooocus confirmed the prompt and negative prompt fields were filled. Open the engine panel and click Generate when ready.", 100, "Engine fields filled. Click Generate when ready.");
        }} else {{
            setStatus("partial-fill", "Engine needs attention", "Fooocus replied, but one or more target fields were not found. Open the engine panel and refresh if needed.", 75, "Engine replied, but field fill was incomplete.");
        }}
    }});
    window.setInterval(watchPlanReady, 600);
}})();
</script>
"""


def send_to_engine_js():
    return f"""
(workflow, fooocus_area, prompt, negative_prompt, setup_steps, next_shots) => {{
    const payload = {{
        type: "fooocus-studio-autofill",
        workflow: workflow || "",
        fooocus_area: fooocus_area || "",
        prompt: prompt || "",
        negative_prompt: negative_prompt || "",
        setup_steps: setup_steps || "",
        next_shots: next_shots || ""
    }};
    if (window.fooocusStudioPostPayloadToEngine) {{
        window.fooocusStudioPostPayloadToEngine(payload);
    }} else {{
        const iframe = document.getElementById("fooocus_engine_iframe");
        if (iframe && iframe.contentWindow) {{
            iframe.contentWindow.postMessage(payload, new URL("{FOOOCUS_ENGINE_URL}").origin);
        }}
    }}
    return [workflow, fooocus_area, prompt, negative_prompt, setup_steps, next_shots];
}}
"""


def build_app():
    with gr.Blocks(title="AI Image Studio", css=CONTROL_UI_CSS) as demo:
        gr.HTML(value=studio_hero_markdown())
        gr.HTML(value=studio_engine_bridge_script())

        with gr.Tab("Studio Control Center"):
            gr.HTML(value=studio_status_panel_html())
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        "## Create the plan\n"
                        "Start here. The raw Fooocus engine stays hidden until you are ready to generate."
                    )
                    with gr.Accordion("Fast no-friction instructions", open=True):
                        gr.Markdown(value=studio_usage_markdown())
                    goal = copyable_textbox(
                        label="What image do you want?",
                        placeholder="Example: realistic full-body resort pool photo, keep my face recognizable, clean luxury lifestyle look",
                        lines=5,
                    )
                    with gr.Accordion("Optional controls", open=False):
                        wants_identity = gr.Checkbox(label="Keep same person or subject", value=True)
                        wants_exact_edit = gr.Checkbox(label="Edit only a specific image or masked area", value=False)
                        wants_bundle = gr.Checkbox(label="Plan a small shot set after the first image", value=False)
                        vram_gb = gr.Slider(label="GPU VRAM in GB", minimum=4, maximum=24, value=6, step=1)
                    plan_btn = gr.Button("Build My Fooocus Plan", variant="primary", elem_id="studio_build_plan_button")

                with gr.Column(scale=1):
                    gr.Markdown("## References\nUpload only what matters. Extra images create churn.")
                    image_1 = gr.Image(label="Reference 1: main subject, source image, or face", type="numpy")
                    image_2 = gr.Image(label="Reference 2: optional mask, style, or support image", type="numpy")
                    image_3 = gr.Image(label="Reference 3: optional pose, layout, or extra angle", type="numpy")

            gr.Markdown(value=copy_controls_summary())
            with gr.Row():
                selected_tool = copyable_textbox(label="Use this Fooocus workflow", interactive=False, elem_id="studio_selected_tool")
                selected_area = copyable_textbox(label="Open this Fooocus tab or area", interactive=False, elem_id="studio_selected_area")

            with gr.Row():
                primary_prompt = copyable_textbox(label="Step 1: Prompt sent to engine", lines=7, elem_id="studio_primary_prompt")
                negative_prompt = copyable_textbox(label="Step 2: Negative prompt sent to engine", lines=7, elem_id="studio_negative_prompt")

            with gr.Row():
                handoff_recipe = copyable_textbox(label="Step 3: Setup steps", lines=10, elem_id="studio_handoff_recipe")
                shot_prompts = copyable_textbox(label="Use these only after the first result", lines=10, elem_id="studio_shot_prompts")

            with gr.Accordion("Review before generating", open=True):
                adapter_preview = gr.Markdown(label="Adapter preview")
                history_preview = gr.Markdown(label="History preview")

            with gr.Accordion("Send to Engine", open=True):
                gr.Markdown(
                    "Click **Send to Engine** after building the plan. This sends the prompt and negative prompt "
                    "to the embedded Fooocus engine automatically through a browser `postMessage` bridge. "
                    "The animated status panel reports only verified Studio/engine handoff states."
                )
                send_to_engine_btn = gr.Button("Send to Engine", variant="primary", elem_id="studio_send_to_engine_button")
                engine_handoff = copyable_textbox(label="Engine send status", lines=10, interactive=False)

            with gr.Row():
                download_prompt_pack_btn = gr.Button("Download Prompt Pack", variant="secondary")
                download_history_btn = gr.Button("Download Session History", variant="secondary")
            with gr.Row():
                prompt_pack_file = gr.File(label="Prompt pack download")
                history_file = gr.File(label="Session history download")

            with gr.Accordion("History gallery and image downloads", open=True):
                gr.Markdown(value=history_gallery_empty_note())
                gr.Gallery(label="Generated image history", value=[], visible=True, columns=4, height=320)
                gr.Markdown(
                    "Image download buttons will appear here after live generation output is connected. "
                    "For now, use the prompt pack and session history downloads."
                )

            with gr.Accordion("Hidden Fooocus engine", open=False):
                gr.Markdown(value=engine_hidden_note())
                gr.Markdown(value=one_ui_note_markdown())
                gr.HTML(value=fooocus_iframe_html())

            with gr.Accordion("Full reasoning", open=False):
                agent_plan = gr.Markdown(label="Agent plan")

            plan_btn.click(
                build_studio_workflow_outputs,
                inputs=[goal, image_1, image_2, image_3, wants_identity, wants_exact_edit, wants_bundle, vram_gb],
                outputs=[agent_plan, primary_prompt, negative_prompt, selected_tool, selected_area, shot_prompts, handoff_recipe, adapter_preview, history_preview],
                queue=False,
            )
            send_to_engine_btn.click(
                build_engine_handoff_text,
                inputs=[selected_tool, selected_area, primary_prompt, negative_prompt, handoff_recipe, shot_prompts],
                outputs=engine_handoff,
                queue=False,
                _js=send_to_engine_js(),
            )
            download_prompt_pack_btn.click(
                write_prompt_pack,
                inputs=[selected_tool, selected_area, primary_prompt, negative_prompt, handoff_recipe, shot_prompts],
                outputs=prompt_pack_file,
                queue=False,
            )
            download_history_btn.click(
                write_history_download,
                inputs=history_preview,
                outputs=history_file,
                queue=False,
            )

        with gr.Tab("Feature Brain"):
            gr.Markdown(
                "## Feature Brain\n"
                "Use this when you want to understand why the Studio chose a Fooocus feature."
            )
            feature_btn = gr.Button("Explain Feature Stack", variant="primary")
            feature_stack = gr.Markdown()
            feature_btn.click(
                explain_feature_stack,
                inputs=[goal, image_1, image_2, image_3, wants_identity, wants_exact_edit, wants_bundle],
                outputs=feature_stack,
                queue=False,
            )
            with gr.Accordion("Full source-of-truth feature catalog", open=False):
                catalog = gr.Markdown(value=feature_catalog_markdown())

        with gr.Tab("Style Coach"):
            gr.Markdown("Use this before generating if you need help choosing a Fooocus style.")
            with gr.Row():
                with gr.Column():
                    style_name = gr.Dropdown(label="Style", choices=STYLE_NAMES, value="Fooocus Photograph")
                    style_goal = copyable_textbox(label="Image goal", value="realistic personal photo", lines=3)
                    explain_btn = gr.Button("Explain Style", variant="primary")
                    recommend_btn = gr.Button("Recommend Styles")
                with gr.Column():
                    style_explanation = gr.Markdown()
                    style_recs = copyable_textbox(label="Recommended styles", lines=8)
            explain_btn.click(explain_style, inputs=[style_name, style_goal], outputs=style_explanation, queue=False)
            recommend_btn.click(recommend_styles, inputs=style_goal, outputs=style_recs, queue=False)

        with gr.Tab("Launcher / Reset"):
            gr.Markdown(value=launcher_controls_markdown())

        with gr.Tab("How to Use"):
            gr.Markdown(
                "## How to use this control UI\n\n"
                "1. Run `START_AI_IMAGE_STUDIO.bat` or `RUN_STUDIO_ONE_UI.bat`.\n"
                "2. Work from `http://127.0.0.1:7872`.\n"
                "3. Use **Studio Control Center** first.\n"
                "4. Click **Build My Fooocus Plan** and watch the status move to Plan ready.\n"
                "5. Click **Send to Engine** to auto-fill the hidden Fooocus prompt fields.\n"
                "6. Open **Hidden Fooocus engine** and confirm the status says Engine fields filled.\n"
                "7. Click **Generate** inside Fooocus. Fooocus shows actual generation progress.\n"
                "8. Review one image, then continue.\n\n"
                "The Studio status panel is accurate for planning and field delivery only; it does not fake diffusion progress."
            )

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7872, inbrowser=True)
