---
### 📄 docs/launch.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نقطة التشغيل الرئيسية (Entry Point) لـ Fooocus. يُحضّر بيئة Python، يُثبّت PyTorch 2.1.0 و Xformers 0.0.23 والمتطلبات من requirements_versions.txt. يُحمّل نماذج VAE التقريبية (xlvaeapp.pth, vaeapp_sd15.pth, xl-to-v1_interposer) ونموذج fooocus_expansion.bin. بعد تجهيز النماذج، يستورد ويُشغّل webui.py.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **prepare_environment()**: تثبيت PyTorch مع CUDA 12.1، Xformers (0.0.23)، والمتطلبات البرمجية. تتحقق من النظام (Windows/Linux) لاختيار طريقة تثبيت xformers المناسبة.
- **ini_args()**: تستورد وتُحلّل معاملات سطر الأوامر من args_manager.py وتُعيد كائن args.
- **download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads)**: تحميل جميع النماذج المطلوبة من HuggingFace. تدعم البديل التلقائي (fallback) للنماذج القديمة إذا لم يكن النموذج الافتراضي متوفراً. تُعيد اسم النموذج الافتراضي وقائمة التحميل.
- **vae_approx_filenames**: قائمة ثابتة بـ 3 ملفات VAE تقريبية مع عناوين URL للتحميل.

المتغيرات العامة: REINSTALL_ALL = False, TRY_INSTALL_XFORMERS = False

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: args_manager.py, fooocus_version.py, build_launcher.py, modules/launch_util.py, modules/model_loader.py, modules/config.py, modules/hash_cache.py, modules/util.py
- **يُستدعى بواسطة**: entry_with_update.py (`from launch import *`)
- **يستورد في النهاية**: webui.py via `from webui import *`
- **الأثر البرمجي**: أي تعديل على طريقة تحميل النماذج أو المتغيرات البيئية سيؤثر على قدرة الأداة على التشغيل

**للمزيد من التفاصيل، راجع:**
- [args_manager.md](args_manager.md) - إدارة معاملات سطر الأوامر
- [entry_with_update.md](entry_with_update.md) - نقطة الدخول مع التحديث
- [build_launcher.md](build_launcher.md) - بناء المشغل
