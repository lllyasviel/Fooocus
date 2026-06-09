---
### 📄 docs/experiments_interrogate.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
سكريبت اختبار (8 أسطر) لميزة وصف الصور (Interrogation). يختبر كلاً من:
1. **BLIP interrogator** (وصف الصور الفوتوغرافية): يقرأ red_box.jpg ويصفها.
2. **WD14 tagger** (وصف الصور الأنمي): يقرأ miku.jpg ويصفها.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **default_interrogator_photo(img)**: من extras.interrogate - وصف BLIP لصورة فوتوغرافية.
- **default_interrogator_anime(img)**: من extras.wd14tagger - وصف WD14 لصورة أنمي.
- **cv2.imread**: قراءة الصور مع عكس القنوات (BGR → RGB).

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: cv2, extras/interrogate.py, extras/wd14tagger.py
- **يُستدعى بواسطة**: لا يُستدعى - سكريبت اختبار مستقل
- **الأثر البرمجي**: لا يؤثر على التطبيق الرئيسي - يُستخدم للتجربة فقط

**للمزيد من التفاصيل، راجع:** [EXTRAS.md](EXTRAS.md)
