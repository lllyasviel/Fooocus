# Task 33 Validation

```powershell
python -m py_compile ai_studio_app.py local_markup\studio_one_ui_note.py
python -m pytest tests/test_ai_studio_app_one_ui_copy.py tests/test_studio_one_ui_note.py -q
```

Full safe suite should include all existing Studio tests plus the new one UI copy tests.
