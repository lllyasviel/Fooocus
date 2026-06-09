---
### 📄 docs/extras/inpaint_mask.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
توليد أقنعة Inpainting باستخدام GroundingDINO للكشف النصي + SAM (Segment Anything Model) للتقطيع الدقيق (أو rembg لإزالة الخلفية). يُحدد الكائنات في الصورة بناءً على برومبت نصي (مثل 'eye', 'face', 'cat') ويولّد قناعاً ثنائياً للتطبيق على Inpaint Worker.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class SAMOptions**: حاوية تكوين لـ GroundingDINO و SAM.
  - المعاملات: dino_prompt, dino_erode_or_dilate, dino_debug, box_threshold, text_threshold, sam_model, sam_max_detections, cloth_category.
- **optimize_masks(masks)**: إزالة المناطق الصغيرة والثقوب من الأقنعة المُتنبأ بها باستخدام remove_small_regions (من segment_anything).
- **generate_mask_from_image(image, sam_options)**: الدالة الرئيسية: تشغيل GroundingDINO للكشف عن الصناديق المُحِدِّدة، تغذية الصناديق إلى SAM للتقطيع، إرجاع القناع الثنائي النهائي (PIL Image) وعدد الاكتشافات.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: modules/config.py, extras/GroundingDINO/util/inference.py (default_groundingdino), extras/sam/predictor.py (SamPredictor), rembg (remove, new_session), segment_anything (sam_model_registry, remove_small_regions)
- **يُستدعى بواسطة**: webui.py (دالة generate_mask), modules/async_worker.py, experiments_mask_generation.py
- **الأثر البرمجي**: تعطيله يُلغي ميزة Generate Mask في واجهة Inpaint
