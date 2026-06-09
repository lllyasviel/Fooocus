---
### 📄 docs/extras/facexlib/detection/__init__.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملف تهيئة حزمة detection في facexlib. يُصدّر دالتين مساعدتين لتحميل نموذج RetinaFace حسب الاسم: init_detection_model (بمسار مطلق) و init_yolo_detection_model (لـ YOLOv5 الافتراضي). يُوفّر واجهة موحدة لاستخدام نماذج كشف الوجوه من بقية أجزاء facexlib.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **init_detection_model(name, half=False, device='cuda')**: إنشاء كاشف RetinaFace: تحميل الأوزان من config.yml + RetinaFace.pth، إنشاء RetinaFace، إرجاع النموذج على الجهاز المطلوب.
- **init_yolo_detection_model(name, half=False, device='cuda')**: (غير مُطبَّق كاملاً) إنشاء كاشف YOLO - يُعيد None حالياً.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/facexlib/detection/retinaface.py (RetinaFace), yaml, torch
- **يُستدعى بواسطة**: extras/facexlib/utils/face_restoration_helper.py (FaceRestoreHelper)
- **الأثر البرمجي**: نقطة الوصول الوحيدة لنماذج كشف الوجوه في facexlib
