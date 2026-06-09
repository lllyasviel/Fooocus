---
### 📄 docs/extras/facexlib/detection/retinaface_utils.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
أدوات مساعدة لـ RetinaFace: توليد مربعات أولية (Prior Boxes) لأحجام مختلفة، فك تشفير توقعات الشبكة إلى صناديق ونقاط معالم حقيقية، وتطبيق NMS (Non-Maximum Suppression) لإزالة الاكتشافات المكررة. يستخدم IOU (Intersection Over Union) لتصفية الصناديق المتداخلة.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class PriorBox**: توليد مربعات أولية (prior/anchor boxes) على خرائط ميزات متعددة المقاييس.
- **decode(loc, priors, variances)**: فك تشفير توقعات الانحدار إلى صناديق إحاطة حقيقية.
- **decode_landm(pre, priors, variances)**: فك تشفير توقعات نقاط المعالم.
- **nms(dets, threshold, force_cpu)**: قمع غير أقصى - إزالة الصناديق ذات IOU > threshold.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: torch, numpy, math
- **يُستدعى بواسطة**: extras/facexlib/detection/retinaface.py (في detect_faces لمعالجة المخرجات)
- **الأثر البرمجي**: ضروري لمعالجة مخرجات الشبكة وتحويلها إلى اكتشافات قابلة للاستخدام
