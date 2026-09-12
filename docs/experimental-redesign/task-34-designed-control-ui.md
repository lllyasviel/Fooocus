# Task 34: Designed Control UI

Branch: `studio/designed-control-ui`

## Goal

Replace the two-page feeling with a cleaner control center:

- Studio controls are the primary UI.
- The raw Fooocus engine is hidden by default.
- Copy-ready text boxes still have copy buttons.
- Prompt pack and session history have download buttons.
- A history gallery area is visible and ready for generated image downloads later.

## Completed

Added:

```text
local_markup/studio_control_ui.py
local_markup/studio_downloads.py
tests/test_studio_control_ui.py
tests/test_studio_downloads.py
tests/test_ai_studio_app_designed_control_ui.py
tests/test_launch_util_temp_cleanup.py
```

Updated:

```text
ai_studio_app.py
tests/test_ai_studio_app_one_ui_copy.py
modules/launch_util.py
```

## Startup Fix

`delete_folder_content()` now recreates the temp folder if it is missing. This prevents startup from crashing when `C:\Users\<user>\AppData\Local\Temp\fooocus` was already removed or never existed.

## User Flow

1. Run `RUN_STUDIO_ONE_UI.bat`.
2. Open `http://127.0.0.1:7872`.
3. Use **Studio Control Center**.
4. Describe the image.
5. Click **Build My Fooocus Plan**.
6. Use copy buttons on the outputs.
7. Download the prompt pack if needed.
8. Open **Hidden Fooocus engine** only when ready to paste and generate.
9. Use **Download Session History** to save the plan/history.

## Runtime Boundary

The engine is visually hidden, not removed. The UI does not directly call the Fooocus worker yet. This is still a safe manual handoff.

## Validation

```powershell
python -m py_compile ai_studio_app.py local_markup\studio_control_ui.py local_markup\studio_downloads.py modules\launch_util.py
python -m pytest tests/test_studio_control_ui.py tests/test_studio_downloads.py tests/test_ai_studio_app_designed_control_ui.py tests/test_ai_studio_app_one_ui_copy.py tests/test_launch_util_temp_cleanup.py -q
```
