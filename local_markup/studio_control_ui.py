from __future__ import annotations


CONTROL_UI_CSS = """
.studio-hero {
    padding: 18px 20px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(20,20,24,0.95), rgba(45,45,58,0.9));
    border: 1px solid rgba(255,255,255,0.12);
}
.studio-card {
    padding: 14px 16px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.04);
}
.studio-small {
    opacity: 0.82;
    font-size: 0.95em;
}
.studio-status-card {
    margin: 14px 0;
    padding: 14px 16px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.05);
}
.studio-status-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
}
.studio-status-dot {
    width: 11px;
    height: 11px;
    border-radius: 999px;
    background: #8a8a8a;
    display: inline-block;
}
.studio-status-dot.active {
    background: #63e6be;
    animation: studioPulse 1.2s infinite ease-in-out;
}
.studio-status-message {
    margin-top: 8px;
    opacity: 0.9;
}
.studio-progress-track {
    margin-top: 12px;
    height: 9px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255,255,255,0.11);
}
.studio-progress-fill {
    height: 100%;
    width: 0%;
    border-radius: 999px;
    background: linear-gradient(90deg, #63e6be, #74c0fc);
    transition: width 260ms ease;
}
.studio-status-steps {
    display: grid;
    grid-template-columns: repeat(4, minmax(120px, 1fr));
    gap: 8px;
    margin-top: 12px;
}
.studio-step {
    padding: 8px 10px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    opacity: 0.72;
    font-size: 0.92em;
}
.studio-step.done {
    opacity: 1;
    background: rgba(99,230,190,0.16);
}
.studio-toast {
    margin-top: 10px;
    padding: 10px 12px;
    border-radius: 12px;
    background: rgba(116,192,252,0.14);
    border: 1px solid rgba(116,192,252,0.25);
    opacity: 0;
    transform: translateY(-4px);
    transition: opacity 220ms ease, transform 220ms ease;
}
.studio-toast.show {
    opacity: 1;
    transform: translateY(0);
}
@keyframes studioPulse {
    0% { box-shadow: 0 0 0 0 rgba(99,230,190,0.5); }
    70% { box-shadow: 0 0 0 8px rgba(99,230,190,0); }
    100% { box-shadow: 0 0 0 0 rgba(99,230,190,0); }
}
"""


def studio_hero_markdown() -> str:
    return (
        "<div class='studio-hero'>"
        "<h1>AI Image Studio Control Center</h1>"
        "<p>Describe the image, send the prepared fields to Fooocus, and generate from one local page.</p>"
        "<p class='studio-small'>No friction: one goal, one first shot, one review cycle.</p>"
        "</div>"
    )


def studio_status_panel_html() -> str:
    return (
        "<div class='studio-status-card' id='studio_status_card'>"
        "<div class='studio-status-title'>"
        "<span class='studio-status-dot' id='studio_status_dot'></span>"
        "<span id='studio_status_title'>Studio ready</span>"
        "</div>"
        "<div class='studio-status-message' id='studio_status_message'>"
        "Build a plan first. Progress here reflects Studio planning and engine-field delivery only; generation progress appears inside Fooocus after you click Generate."
        "</div>"
        "<div class='studio-progress-track'><div class='studio-progress-fill' id='studio_progress_fill'></div></div>"
        "<div class='studio-status-steps'>"
        "<div class='studio-step' id='studio_step_plan'>1. Plan</div>"
        "<div class='studio-step' id='studio_step_send'>2. Send</div>"
        "<div class='studio-step' id='studio_step_ack'>3. Engine filled</div>"
        "<div class='studio-step' id='studio_step_generate'>4. Generate manually</div>"
        "</div>"
        "<div class='studio-toast' id='studio_toast'>Status updates will appear here.</div>"
        "</div>"
    )


def launcher_controls_markdown() -> str:
    return (
        "## Launcher and reset controls\n\n"
        "Use the new launcher menu when you want fewer commands:\n\n"
        "```powershell\n"
        ".\\START_AI_IMAGE_STUDIO.bat\n"
        "```\n\n"
        "### What the reset options mean\n\n"
        "- **Refresh** opens the Studio page again without stopping anything.\n"
        "- **Hot reset** restarts only the AI Studio page on port `7872`; the Fooocus engine on `7865` stays warm.\n"
        "- **Cold reset** stops both Studio and Fooocus local ports, clears the local temp session folder, then starts clean.\n\n"
        "### Where crashes are logged\n\n"
        "- `logs/studio/latest-ai-studio.log`\n"
        "- `logs/studio/latest-fooocus-engine.log`\n\n"
        "Generation progress is still shown by Fooocus itself after you click **Generate**."
    )


def engine_hidden_note() -> str:
    return (
        "## Engine is hidden by default\n\n"
        "Use **Send to Engine** first. Open the engine panel to confirm the prompt fields are filled and then click Generate."
    )


def history_gallery_empty_note() -> str:
    return (
        "## History Gallery\n\n"
        "Image downloads appear here after generated outputs are connected. For now, use the prompt pack and session history downloads."
    )
