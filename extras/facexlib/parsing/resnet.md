---
### 📄 docs/extras/facexlib/parsing/resnet.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ResNet معدّل يستخدم كـ backbone لنموذج BiSeNet. يُنفّذ ResNet18 مخصصاً مع BatchNorm و ReLU، يحافظ على خرائط الميزات بأحجام مختلفة (C1: 64, C2: 64, C3: 128, C4: 256) للتغذية إلى مسار السياق في BiSeNet.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class BasicBlock(nn.Module)**: كتلة ResNet أساسية (Conv3×3 + BN + ReLU + Conv3×3 + BN + Residual).
- **class ResNet18(nn.Module)**: ResNet18 كامل: Conv أولي (7×7) + MaxPool + 4 طبقات BasicBlock (2,2,2,2).
  - `forward(self, x)`: إرجاع قاموس {'c1', 'c2', 'c3', 'c4'} بمقاييس مختلفة.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: torch, torch.nn
- **يُستدعى بواسطة**: extras/facexlib/parsing/bisenet.py (ContextPath)
- **الأثر البرمجي**: أساسي لـ BiSeNet
