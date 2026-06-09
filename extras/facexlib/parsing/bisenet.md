---
### 📄 docs/extras/facexlib/parsing/bisenet.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج BiSeNet (Bilateral Segmentation Network) لتجزئة الوجه. يُصنّف كل بكسل في الصورة إلى فئة من 19 فئة (الجلد، الحاجب الأيسر، الحاجب الأيمن، العين اليسرى، العين اليمنى، الأنف، الفم، الشفة العليا، الشفة السفلى، الشعر، إلخ). يستخدم ResNet18 كـ backbone مع مسارين: Spatial Path + Context Path.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class ConvBNReLU(nn.Module)**: Conv2d + BatchNorm + ReLU.
- **class ResNet18(nn.Module)**: ResNet18 معدّل لتجزئة الصور.
- **class SpatialPath(nn.Module)**: المسار المكاني - 3 طبقات Conv مع stride=2 للحفاظ على الدقة المكانية.
- **class AttentionRefinementModule(nn.Module)**: وحدة تحسين الانتباه مع Global Average Pooling.
- **class ContextPath(nn.Module)**: مسار السياق - ResNet18 + ARM لجمع السياق العام.
- **class FeatureFusionModule(nn.Module)**: دمج الميزات المكانية والسياقية مع انتباه قنوات.
- **class BiSeNet(nn.Module)**: النموذج الكامل: SpatialPath + ContextPath + FeatureFusionModule + رأس الإخراج (3×Conv لإنتاج 19 قناة فئة).

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: torch, torch.nn, functools, extras/facexlib/parsing/resnet.py
- **يُستدعى بواسطة**: extras/facexlib/parsing/__init__.py
- **الأثر البرمجي**: أساسي لتجزئة أجزاء الوجه
