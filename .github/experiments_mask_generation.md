---
### 📄 docs/experiments_mask_generation.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
سكريبت اختبار (24 سطر) لتوليد قناع Inpaint باستخدام GroundingDINO + SAM. يقرأ صورة 'cat.webp'، يُنشئ SAMOptions مع dino_prompt='eye' (كشف العين)، ويُولّد قناعاً باستخدام generate_mask_from_image() من extras.inpaint_mask. يعرض القناع الناتج.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **SAMOptions**: كائن تكوين من extras.inpaint_mask - يحدد معاملات الكشف: dino_prompt='eye', box_threshold=0.3, text_threshold=0.25, model_type='vit_b', max_detections=2.
- **generate_mask_from_image(image, sam_options=sam_options)**: دالة توليد القناع - تُعيد (mask_image, ..., ..., ...).
- **Image.open / Image.fromarray / merged_masks_img.show()**: قراءة الصورة، تحويل النتيجة إلى PIL، وعرضها.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: numpy, PIL, extras/inpaint_mask.py (SAMOptions, generate_mask_from_image)
- **يُستدعى بواسطة**: لا يُستدعى - سكريبت اختبار مستقل
- **الأثر البرمجي**: لا يؤثر على التطبيق الرئيسي - يُستخدم للتجربة فقط
- **مرجع**: https://github.com/sail-sg/EditAnything (مذكور في التعليق الأول)

**للمزيد من التفاصيل، راجع:** [EXTRAS.md](EXTRAS.md)
