# وثيقة ملف: samplers.md
## 1. الوظيفة الأساسية للملف
تنفيذ عينات CFG (Classifier-Free Guidance) واستدعاء المخططات (Schedulers) المختلفة لعملية إزالة الضوضاء في نماذج الانتشار.

## 2. الدوال والكلاسات (Functions & Classes)
- `sample()`: الدالة الرئيسية لأخذ العينات
- `CFGDenoiser()`: كلاس لازالة الضوضاء مع التوجيه بدون مصنف
- `KSAMPLER()`: كلاس لخوارزمية K-Sampling
- `BasicScheduler()`: مخطط أساسي لإزالة الضوضاء
- `SDSampler()`: مخطط عينات Stable Diffusion
- `add_area()`: إضافة منطقة للعينات
- `calc_cond()`: حساب الشرط
- `encode_model_conds()`: ترميز شروط النموذج

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف: `sample.py`, `ldm_patched.py`
- الملفات التي يستدعيها هذا الملف لتشغيله: `model_base.py`, `model_patcher.py`, `conds.py`
