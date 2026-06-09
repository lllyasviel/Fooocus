---
### 📄 docs/extras/facexlib/parsing/parsenet.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج ParseNet (Face Parsing Network) بديل لـ BiSeNet. يُنفّذ شبكة تجزئة أبسط مع ResNet bottleneck blocks. يُصنّف كل بكسل إلى فئة من الـ 19 فئة. يُستخدم كبديل (fallback) إذا لم يكن BiSeNet متاحاً.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class Bottleneck(nn.Module)**: كتلة ResNet bottleneck (Conv1×1 → Conv3×3 → Conv1×1).
- **class ResNet(nn.Module)**: ResNet معدّل مع طبقات Conv أولية + 4 طبقات من Bottleneck blocks.
- **class ParseNet(nn.Module)**: النموذج الكامل: ResNet backbone + Conv2d نهائي لإنتاج 19 قناة فئة.
  - `forward(self, x)`: تشغيل الصورة عبر backbone ثم Conv2d، إرجاع خريطة التجزئة.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: torch, torch.nn
- **يُستدعى بواسطة**: extras/facexlib/parsing/__init__.py
- **الأثر البرمجي**: بديل لتجزئة الوجه عند عدم توفر BiSeNet
