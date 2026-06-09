---
### 📄 docs/extras/resampler.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
تنفيذ Perceiver Resampler لتقليل عدد رموز ميزات الصورة من CLIP (257 رمزاً) إلى عدد ثابت من الاستعلامات القابلة للتعلم (16 استعلاماً). يُستخدم في IP-Adapter Plus لتحسين كفاءة ربط ميزات الصورة بـ Cross-Attention في UNet.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class FeedForward(nn.Module)**: شبكة تغذية أمامية مع LayerNorm + Linear + GELU + Dropout + Linear.
- **reshape_tensor(x, heads)**: إعادة تشكيل tensor من (batch, length, width) إلى (batch, heads, length, dim_per_head) للانتباه متعدد الرؤوس.
- **class PerceiverAttention(nn.Module)**: انتباه تبادلي (cross-attention): الاستعلامات الكامنة (latents) تنتبه إلى مفاتيح/قيم الصورة+الكامنة المدمجة.
  - `forward(self, x, latents)`: حساب attention مع RoPE وتطبيع.
- **class Resampler(nn.Module)**: كامل Perceiver Resampler: استعلامات كامنة قابلة للتعلم (nn.Parameter)، طبقات إسقاط (projection)، وتكرار 4 طبقات من Attention+FFN.
  - `forward(self, x)`: إسقاط المدخلات عبر proj_in، تشغيل عبر طبقات الانتباه/FFN، إرجاع التضمينات المُعاد أخذ العينات.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: torch, torch.nn, math
- **يُستدعى بواسطة**: extras/ip_adapter.py
- **الأثر البرمجي**: تعطيله يُعطّل IP-Adapter Plus
