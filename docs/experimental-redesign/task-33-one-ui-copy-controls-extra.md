# Task 33: One UI Copy Controls

## What this adds

- Every copy-ready text box uses Gradio copy buttons.
- AI Studio has a **Fooocus Engine** tab with the local Fooocus page embedded.
- A one-command Windows launcher starts both local services.
- Tests lock down the one UI and copy-control behavior.

## User flow

1. Run `RUN_STUDIO_ONE_UI.bat`.
2. Open `http://127.0.0.1:7872`.
3. Use **Studio Agent** to build the plan.
4. Use the copy buttons on the prompt, negative prompt, workflow, and setup boxes.
5. Open **Fooocus Engine** inside the same UI.
6. Paste and generate one image first.

## Boundary

This is still manual handoff. It does not call the Fooocus worker directly.
