# وثيقة ملف: sampling.py
## 1. الوظيفة الأساسية للملف
يحتوي على خوارزميات Sampling المختلفة لـ K-diffusion مثل Karras وExponential وVP.

## 2. الدوال والكلاسات (Functions & Classes)
- append_zero(): دالة لإضافة صفر إلى نهاية المصفوفة
- get_sigmas_karras(): دالة لبناء جدولة ضوضاء Karras
- get_sigmas_exponential(): دالة لبناء جدولة ضوضاء أسية
- get_sigmas_polyexponential(): دالة لبناء جدولة ضوجاء متعددة الحدود في log sigma
- get_sigmas_vp(): دالة لبناء جدولة ضوضاء VP مستمرة
- to_d(): دالة لتحويل إخراج الـ denoiser إلى مشتق Karras ODE
- get_ancestral_step(): دالة للحصول على خطوة أنسابية
- sample_euler(): دالة لـ sampling باستخدام خوارزمية Euler
- sample_euler_ancestral(): دالة لـ sampling باستخدام خوارزمية Euler Ancestral
- sample_heun(): دالة لـ sampling باستخدام خوارزمية Heun
- sample_dpm_2(): دالة لـ sampling باستخدام خوارزمية DPM-2
- sample_dpm_2_ancestral(): دالة لـ sampling باستخدام خوارزمية DPM-2 Ancestral
- sample_lms(): دالة لـ sampling باستخدام خوارزمية LMS
- sample_dpm_fast(): دالة لـ sampling باستخدام خوارزمية DPM Fast
- sample_dpm_adaptive(): دالة لـ sampling باستخدام خوارزمية DPM Adaptive
- sample_dpmpp_2s_ancestral(): دالة لـ sampling باستخدام خوارزمية DPM++ 2S Ancestral
- sample_dpmpp_sde(): دالة لـ sampling باستخدام خوارزمية DPM++ SDE
- sample_dpmpp_2m(): دالة لـ sampling باستخدام خوارزمية DPM++ 2M
- sample_dpmpp_2m_sde(): دالة لـ sampling باستخدام خوارزمية DPM++ 2M SDE
- sample_dpmpp_3m_sde(): دالة لـ sampling باستخدام خوارزمية DPM++ 3M SDE
- sample_ddim(): دالة لـ sampling باستخدام خوارزمية DDIM
- sample_ddim_ve(): دالة لـ sampling باستخدام خوارزمية DDIM VE

## 3. الاعتماديات والعلاقات
- الملفات التي تستدعي هذا الملف:
  - ldm_patched/contrib/external_custom_sampler.py
  - ldm_patched/modules/samplers.py
- الملفات التي يستدعيها هذا الملف لتشغيله:
  - math
  - scipy.integrate
  - torch
  - torch.nn
  - torchsde
  - tqdm
  - ldm_patched/k_diffusion/utils.py