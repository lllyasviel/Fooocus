---
### 📄 مسار الملف الافتراضي: docs/root/launch.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملف التشغيل الرئيسي (Entry Point) لأداة Fooocus. مسؤول عن إعداد بيئة التشغيل وتثبيت المتطلبات البرمجية وتحميل النماذج الأولية وتهيئة الإعدادات ثم إطلاق واجهة المستخدم الرسومية.

المهام الرئيسية:
- تعيين المتغيرات البيئية (مثل PYTORCH_ENABLE_MPS_FALLBACK و GRADIO_SERVER_PORT)
- تثبيت PyTorch و Xformers و المتطلبات من ملف requirements_versions.txt
- تحميل نماذج VAE التقريبية (xlvaeapp.pth, vaeapp_sd15.pth, xl-to-v1_interposer)
- تحميل نموذج fooocus_expansion.bin لتوسيع البرومبت
- تحميل النماذج الافتراضية (checkpoints, embeddings, loras, vae) مع دعم البديل التلقائي للنماذج القديمة
- تهيئة كاش التوقيعات (hash_cache) للنماذج

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **prepare_environment()**: تثبيت PyTorch (cu121) و Xformers (0.0.23) و المتطلبات البرمجية. تحقق من النظام (Windows/Linux) لاختيار طريقة التثبيت المناسبة.
- **ini_args():** تستورد وتعيد كائن Args من args_manager.
- **download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads)**: تحميل جميع النماذج المطلوبة. تدعم البديل التلقائي (fallback) للنماذج القديمة إذا لم تكن النموذج الافتراضي متوفر.
- **vae_approx_filenames**: قائمة ثابتة بملفات VAE التقريبية وعناوين URL للتحميل من HuggingFace.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: args_manager.py, fooocus_version.py, build_launcher.py, modules/launch_util.py, modules/model_loader.py, modules/config.py, modules/hash_cache.py, modules/util.py, webui.py
- **يُستدعى بواسطة**: entry_with_update.py (النقطة الأولية الفعلية)
- **الأثر البرمجي**: أي تعديل على طريقة تحميل النماذج أو المتغيرات البيئية سيؤثر مباشرة على قدرة الأداة على التشغيل
