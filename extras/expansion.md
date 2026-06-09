---
### 📄 docs/extras/expansion.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
توسيع البرومبت (Prompt Expansion) باستخدام نموذج GPT-2 مخصص (Fooocus V2 Expansion). يأخذ مُدخلاً نصياً قصيراً ويولّد امتدادات وصفية باستخدام قائمة كلمات إيجابية محددة (positive_words) مع تحيز logits لتوجيه التوليد. يُحمَّل النموذج مع logits_processor يمنع تكرار الكلمات ويُرجّح الكلمات الإيجابية.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **safe_str(text)**: إزالة المسافات الزائدة وتقليم علامات الترقيم من النص.
- **remove_pattern(text, pattern)**: إزالة أنماط محددة من النص.
- **class FooocusExpansion**: محرك توسيع البرومبت GPT-2.
  - `__init__(self)`: تحميل tokenizer، قائمة positive_words (150+ كلمة إيجابية)، ونموذج GPT-2 مع logits bias ثنائي القطب.
  - `logits_processor(self, input_ids, scores)`: معالج logits مخصص: يُرجّح الكلمات من positive_words (+100) ويُخفّف الكلمات المولّدة سابقاً (-100).
  - `__call__(self, prompt, seed)`: توسيع البرومبت عبر GPT-2 generation مع constrained decoding. يُعيد النص الموسّع.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: ldm_patched/modules/model_management.py, modules/config.py, ldm_patched/modules/model_patcher.py, transformers (AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, set_seed)
- **يُستدعى بواسطة**: modules/default_pipeline.py, modules/async_worker.py
- **الأثر البرمجي**: تعطيل التوسيع يُضعف جودة البرومبتات القصيرة وقد يُقلّل من تنوع المخرجات
