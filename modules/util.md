# وثيقة ملف: util.md
## 1. الوظيفة الأساسية للملف
 أدوات عامة متنوعة. يوفر دوال لمعالجة الصور، حساب التجزئة، تحويل التنسيقات، ومعالجة النصوص.

## 2. الدوال والكلاسات (Functions & Classes)
- **resample_image()**: إعادة تsample الصورة إلى حجم جديد.
- **resize_image()**: تغيير حجم الصورة بأوضاع مختلفة (0: resize, 1: crop+resize, 2: fill+resize).
- **erode_or_dilate()**: تطبيق erosion أو dilation على القناع.
- **sha256()**: حساب تجزئة SHA256 لملف.
- **get_file_from_folder_list()**: البحث عن ملف في قائمة مجلدات.
- **get_enabled_loras()**: استخراج LoRA الممكّنة من القائمة.
- **generate_temp_filename()**: إنشاء اسم ملف مؤقت معطياً.
- **LORAS_PROMPT_PATTERN**: نمط regex لاستخراج LoRA من النص.

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: معظم ملفات modules/
- الملفات التي يستدعيها هذا الملف لتشغيله: modules/config.py, modules/sdxl_styles.py, modules/flags.py, cv2, PIL, hashlib