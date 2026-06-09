# وثيقة ملف: external_model_merging.py
## 1. الوظيفة الأساسية للملف
يحتوي على عقد متخصصة في دمج ودمج نماذج التوليد المختلفة.

## 2. الدوال والكلاسات (Functions & Classes)
- ModelMergeSimple: كلاس لدمج نموذجين ببساطة باستخدام ratio
- ModelSubtract: كلاس لطرح نموذجين
- ModelAdd: كلاس لجمع نموذجين
- ModelMultiply: كلاس لضرب نموذجين
- ModelMergeBlocks: كلاس لدمج نماذج على مستوى الكتل
- ModelMergeSD1: كلاس لدمج نماذج SD1 على مستوى الكتل
- ModelMergeSD2: كلاس لدمج نماذج SD2 على مستوى الكتل
- ModelMergeSDXL: كلاس لدمج نماذج SDXL على مستوى الكتل

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/contrib/external_video_model.py
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - ldm_patched/modules/sd.py
  - ldm_patched/modules/utils.py
  - ldm_patched/modules/model_base.py
  - ldm_patched/modules/model_management.py
  - ldm_patched/utils/path_utils.py
  - ldm_patched/modules/args_parser.py
  - json
  - os