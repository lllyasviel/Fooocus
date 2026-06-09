# وثيقة ملف: external_model_advanced.py
## 1. الوظيفة الأساسية للملف
يحتوي على إعدادات النموذج المتقدمة مثل LCM وModelSamplingDiscreteDistilled وRescaleZeroTerminalSNR.

## 2. الدوال والكلاسات (Functions & Classes)
- LCM: كلاس لتطبيق Latent Consistency Models مع معادلة حسابية مخصصة
- ModelSamplingDiscreteDistilled: كلاس للعينات المتقطعة المقتصة
- rescale_zero_terminal_snr_sigmas(): دالة لإعادة تقييم sigmas بterminal SNR صفر
- ModelSamplingContinuousEDM: كلاس للعينات المستمرة EDM
- ModelSamplingContinuousV: كلاس للعينات المستمرة V
- SamplerCustom: كلاس لـ sampling مخصص مع scheduler وsampler محددين
- SamplerCustomAdvanced: كلاس لـ sampling مخصص متقدم
- KSamplerAdvanced: كلاس لـ sampling متقدم باستخدام KSampler

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - ldm_patched/utils/path_utils.py
  - ldm_patched/modules/sd.py
  - ldm_patched/modules/model_sampling.py
  - torch