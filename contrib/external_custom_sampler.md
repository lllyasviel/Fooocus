# وثيقة ملف: external_custom_sampler.py
## 1. الوظيفة الأساسية للملف
يحتوي على عقد متخصصة في الجدولة والم징عات المخصصة للـ sampling، بما في ذلك Karras وExponential وغيرها.

## 2. الدوال والكلاسات (Functions & Classes)
- BasicScheduler: كلاس للجدولة الأساسية باستخدام جدولة نموذج SAMPLER_NAMES
- KarrasScheduler: كلاس لجدولة Karras مع إعدادات sigma_max وsigma_min
- ExponentialScheduler: كلاس للجدولة الأسية
- BetaScheduler: كلاس للجدولة بيتا
- LCMScheduler: كلاس لجدولة LCM (Latent Consistency Models)
- DPMPP_SDE Scheduler: كلاس لجدولة DPMPP-SDE
- UniformSampler: كلاس لم_sample موحد
- NormalDistributionSampler: كلاس لم_sample بالتوزيع الطبيعي
- CFGNormalizer: كلاس لتطبيع CFG
- KSampler: كلاس لـ sampling باستخدام خوارزميات K-diffusion

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - غير معروف (يُستخدم كعقدة منفصلة)
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - ldm_patched/modules/samplers.py
  - ldm_patched/modules/sample.py
  - ldm_patched/k_diffusion/sampling.py
  - ldm_patched/utils/latent_visualization.py
  - torch
  - ldm_patched/modules/utils.py