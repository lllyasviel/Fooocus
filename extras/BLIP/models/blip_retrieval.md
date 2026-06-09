---
### 📄 docs/extras/BLIP/models/blip_retrieval.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
نموذج استرجاع الصور-النصوص (Image-Text Retrieval) من BLIP. يُحسّن التمثيلات المشتركة للصور والنصوص باستخدام فقدان ITC (متباين) + ITM (توافقي) لاسترجاع أفضل النصوص المطابقة للصورة والعكس.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class BLIP_Retrieval(nn.Module)**: نموذج الاسترجاع.
  - `forward(self, image, caption, alpha)`: حساب خسائر ITC و ITM مع وزن alpha.
- **blip_retrieval(pretrained, image_size, vit, vit_grad_ckpt, vit_ckpt_layer, med_config)**: إنشاء النموذج مع تحميل أوزان pretrained.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/BLIP/models/blip.py, extras/BLIP/models/med.py
- **يُستدعى بواسطة**: غير مستخدم مباشرة في Fooocus
- **الأثر البرمجي**: غير مؤثر - نموذج احتياطي من BLIP
