---
### 📄 docs/experiments_face.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
سكريبت اختبار (7 أسطر) لميزة قص الوجه. يقرأ صورة 'lena.png' باستخدام OpenCV، يُمرّرها إلى cropper.crop_image() من extras.face_crop، ويحفظ النتيجة في 'lena_result.png'.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **cropper.crop_image(img)**: دالة من extras.face_crop - تكشف الوجوه وتُعيد قصاً مُحاذياً لأكبر وجه.
- **cv2.imread/cv2.imwrite**: قراءة وكتابة الصور باستخدام OpenCV.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: cv2 (OpenCV), extras/face_crop.py
- **يُستدعى بواسطة**: لا يُستدعى - سكريبت اختبار مستقل
- **الأثر البرمجي**: لا يؤثر على التطبيق الرئيسي - يُستخدم للتجربة فقط

**للمزيد من التفاصيل، راجع:** [EXTRAS.md](EXTRAS.md)
