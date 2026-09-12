# Task 33 Full Validation

Run after checking out `studio/one-ui-copy-controls`:

```powershell
python -m py_compile ai_studio_app.py local_markup\studio_one_ui_note.py
python -m pytest tests/test_ai_studio_app_one_ui_copy.py tests/test_studio_one_ui_note.py tests/test_one_ui_launcher_exists.py -q
```

Then start the app:

```powershell
.\RUN_STUDIO_ONE_UI.bat
```

Use one page:

```text
http://127.0.0.1:7872
```
