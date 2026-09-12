# Task 33: One UI Copy Controls

Branch: `studio/one-ui-copy-controls`

## Goal

Make the Studio usable from one browser page with no copy friction.

## Completed

- Added copy buttons to every copy-ready textbox through `copyable_textbox()`.
- Added a **Fooocus Engine** tab inside AI Studio using a local iframe.
- Added a one-command Windows launcher: `RUN_STUDIO_ONE_UI.bat`.
- Added one UI helper text and tests.

## User flow

1. Run `RUN_STUDIO_ONE_UI.bat`.
2. Open `http://127.0.0.1:7872`.
3. Use **Studio Agent** to build the plan.
4. Click the copy button on the prompt and negative prompt boxes.
5. Open **Fooocus Engine** in the same UI.
6. Paste into Fooocus and generate one image first.

## Validation

```powershell
python -m py_compile ai_studio_app.py local_markup\studio_one_ui_note.py
python -m pytest tests/test_ai_studio_app_one_ui_copy.py tests/test_studio_one_ui_note.py -q
```

## Runtime Boundary

This is still a manual handoff. It does not call the Fooocus worker directly.
