---
### 📄 docs/extras/facexlib/detection/matlab_cp2tform.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
محاكاة دالة MATLAB's cp2tform (Control Point to Transformation) في Python. يُحسب التحويل الأفيني (2D affine) من نقاط تحكم مصدرية إلى نقاط مرجعية باستخدام حل المربعات الصغرى (Least Squares). يُستخدم لحساب مصفوفة المحاذاة الدقيقة لتوحيد أوضاع الوجوه.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **mstruct_to_shapesize(mstruct)**: تحويل بنية MATLAB إلى أبعاد الشكل.
- **findNonreflectiveSimilarity(uv, xy, K, M, T)**: حساب تشابه غير عاكس (similarity without reflection) بين نقطتين.
- **findSimilarity(uv, xy, K, M, T)**: حساب التشابه مع إمكانية الانعكاس.
- **getInvTransform(mstruct)**: حساب التحويل العكسي.
- **cubic_exception(uv, xy, K, M, T)**: الاستيفاء التكعيبي للحالات الاستثنائية.
- **cp2tform(uv, xy, transform_type)**: الدالة الرئيسية - حساب التحويل بين مجموعتي نقاط تحكم.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: numpy, math
- **يُستدعى بواسطة**: extras/facexlib/detection/align_trans.py
- **الأثر البرمجي**: أساسي لحساب مصفوفات التحويل الأفيني في محاذاة الوجه
