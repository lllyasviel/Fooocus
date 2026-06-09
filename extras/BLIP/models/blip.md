---
### 📄 docs/extras/BLIP/models/blip.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
يُعرّف النماذج الأساسية لمكتبة BLIP: BLIP_Base لاستخراج السمات المشتركة بين الصور والنصوص، و BLIP_Decoder لتوليد التسميات التوضيحية (captions). يُنشئ النماذج مع أوزان مسبقة التدريب (pretrained) من مسارات محددة في config.py. النموذج المُستخدَم فعلياً في Fooocus هو blip_decoder (للوصف).

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class BLIP_Base(nn.Module)**: نموذج أساسي لاستخراج سمات الصور والنصوص والمتعددة الوسائط عبر Vision Transformer ومشفر BERT.
- **class BLIP_Decoder(nn.Module)**: نموذج فك تشفير لتوليد التسميات التوضيحية للصور باستخدام Cross-Attention بين الصورة والنص.
- **blip_decoder(pretrained, image_size, vit, vit_grad_ckpt, vit_ckpt_layer, prompt, med_config)**: إنشاء BLIP_Decoder مع تحميل أوزان pretrained.
- **blip_feature_extractor(pretrained, image_size, vit, vit_grad_ckpt, vit_ckpt_layer, prompt, med_config)**: إنشاء BLIP_Base مع تحميل أوزان pretrained.
- **init_tokenizer()**: تهيئة BertTokenizer من transformers مع إضافة رموز خاصة [DEC], [ENC], [SEP1], [SEP2].
- **create_vit(vit, image_size, use_grad_checkpointing, ckpt_layer, drop_path_rate)**: إنشاء Vision Transformer (ViT-B/16 أو ViT-L/14) مع grad checkpointing.
- **load_checkpoint(model, checkpoint_path)**: تحميل أوزان النموذج من checkpoint.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: extras/BLIP/models/vit.py (create_vit), extras/BLIP/models/med.py (BertEncoder, BertLMHeadModel), transformers (BertTokenizer)
- **يُستدعى بواسطة**: extras/interrogate.py (blip_decoder), extras/BLIP/models/blip_itm.py, extras/BLIP/models/blip_nlvr.py, extras/BLIP/models/blip_pretrain.py, extras/BLIP/models/blip_retrieval.py, extras/BLIP/models/blip_vqa.py
- **الأثر البرمجي**: أساسي لميزة وصف الصور (Interrogate) في Fooocus
