---
### 📄 docs/args_manager.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
إدارة معاملات سطر الأوامر (CLI Arguments) الخاصة بـ Fooocus. يوسّع محلل المعاملات الأساسي من ldm_patched بإضافة 15+ معامل خاص بـ Fooocus: --share، --preset، --language، --theme، --disable-image-log، --disable-metadata، --always-download-new-model، --rebuild-hash-cache، وغيرها.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
لا يحتوي على دوال أو كلاسات - كود في المستوى العلوي:
- **args_parser.parser.add_argument(...)**: 15 إضافة لمعاملات سطر الأوامر.
- **args_parser.parser.set_defaults(...)**: تعيين القيم الافتراضية (disable_cuda_malloc=True, in_browser=True, port=None).
- **args_parser.args = args_parser.parser.parse_args()**: تحليل المعاملات.
- **args_parser.args.always_offload_from_vram**: تعيين عكسي لـ disable_offload_from_vram.

المعاملات المُضافة:
| المعامل | النوع | الوصف |
|---------|-------|-------|
| --share | bool | مشاركة على Gradio |
| --preset | str | تطبيق قالب UI |
| --disable-preset-selection | bool | تعطيل اختيار القالب |
| --language | str | ترجمة UI من مجلد language/ |
| --disable-offload-from-vram | bool | إجبار تحميل النماذج إلى VRAM |
| --theme | str | سمة الواجهة (فاتح/داكن) |
| --disable-image-log | bool | منع حفظ الصور والسجلات |
| --disable-analytics | bool | تعطيل تحليلات Gradio |
| --disable-metadata | bool | تعطيل حفظ البيانات الوصفية |
| --disable-preset-download | bool | تعطيل تحميل نماذج القالب |
| --disable-enhance-output-sorting | bool | تعطيل ترتيب مخرجات التحسين |
| --enable-auto-describe-image | bool | تفعيل وصف الصورة التلقائي |
| --always-download-new-model | bool | إجبار تحميل النماذج الجديدة |
| --rebuild-hash-cache | int | إعادة بناء كاش التوقيعات |

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: ldm_patched/modules/args_parser.py (المحلل الأساسي)
- **يُستدعى بواسطة**: launch.py (`from args_manager import args`), webui.py, modules/config.py, modules/async_worker.py, modules/auth.py, modules/patch.py, extras/inpaint_mask.py
- **الأثر البرمجي**: أي معامل جديد يُضاف هنا يصبح متاحاً كمتغير args في جميع أنحاء التطبيق
