---
### 📄 docs/extras/facexlib/detection/retinaface_net.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
يُعرّف البنية الشبكية (network architecture) لـ RetinaFace: FPN (Feature Pyramid Network) مع SSH (Single Stage Headless)، طبقات الكشف (Classification + BBox Regression + Landmark Regression)، والـ backbone (MobileNet أو ResNet).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class ConvDPUnit(nn.Module)**: وحدة التفاف مع Depthwise + Pointwise.
- **class ConvUnit(nn.Module)**: وحدة التفاف مع BatchNorm + PReLU.
- **class MobileNetV1(nn.Module)**: شبكة أساسية MobileNetV1.
- **class FPN(nn.Module)**: شبكة هرم الميزات مع 3 مستويات (C3, C4, C5 → P3, P4, P5) مع Conv2d 1×1.
- **class SSH(nn.Module)**: وحدات كشف للمقاييس المتعددة (P3, P4, P5) مع ContextModule و Conv2d 3×3.
- **class ClassHead(nn.Module)**: رأس تصنيف - Conv2d ينتج قناتين (وجه/خلفية).
- **class BboxHead(nn.Module)**: رأس انحدار الصندوق - Conv2d ينتج 4 قنوات (x, y, w, h).
- **class LandmarkHead(nn.Module)**: رأس نقاط المعالم - Conv2d ينتج 10 قنوات (5 نقاط × 2).
- **class RetinaFaceMobileNet(nn.Module)**: RetinaFace الكامل مع MobileNet backbone.
- **class RetinaFaceResNet50(nn.Module)**: RetinaFace الكامل مع ResNet50 backbone.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: torch, torch.nn, math
- **يُستدعى بواسطة**: extras/facexlib/detection/retinaface.py (RetinaFace)
- **الأثر البرمجي**: البنية الأساسية لنموذج كشف الوجوه
