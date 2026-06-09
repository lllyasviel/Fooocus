---
### 📄 docs/shared.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملف مشارك بسيط (سطر واحد) يحتوي على المتغير العام `gradio_root = None`. يُستخدم لتخزين كائن gr.Blocks الرئيسي لـ Gradio بعد إنشائه في webui.py، ليكون متاحاً من أي مكان في المشروع (خاصة async_worker.py).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **gradio_root**: متغير عام من نوع gr.Blocks. يُعيَّن في webui.py (السطر 153) عبر `shared.gradio_root = gr.Blocks(title=title).queue()`.
  - يُستخدم في async_worker.py لطباعة معلومات الاتصال (URL المحلي والخارجي).
  - يُستخدم في webui.py لربط أحداث التحميل (load events) وعناصر UI.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **لا يستدعي أي ملفات أخرى**
- **يُستدعى بواسطة**: webui.py (يُعيّنه), modules/async_worker.py (يقرؤه), modules/localization.py
- **الأثر البرمجي**: إزالة هذا المتغير ستكسر الاتصال بين webui.py و async_worker.py
