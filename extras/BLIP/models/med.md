---
### 📄 docs/extras/BLIP/models/med.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
تنفيذ BERT مخصص (Modified Encoder-Decoder) للمكتبة BLIP. يوفر مشفر BERT كامل مع طبقات Self-Attention، Cross-Attention (للتشفير-فك التشفير)، ورؤوس تنبؤ للكلمات المفقودة (MLM). يُستخدم كقاعدة لجميع نماذج BLIP الأخرى.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **class BertEmbeddings(nn.Module)**: تضمين الكلمات + المواقع + نوع الجملة (token/segment/position).
- **class BertSelfAttention(nn.Module)**: انتباه ذاتي متعدد الرؤوس مع Dropout.
- **class BertSelfOutput(nn.Module)**: إخراج الانتباه الذاتي مع LayerNorm + Dropout + Residual.
- **class BertAttention(nn.Module)**: يدمج SelfAttention + SelfOutput.
- **class BertIntermediate(nn.Module)**: طبقة متوسطة (Linear + GELU).
- **class BertOutput(nn.Module)**: إخراج الطبقة مع LayerNorm + Dropout + Residual.
- **class BertLayer(nn.Module)**: طبقة BERT كاملة (Attention + Intermediate + Output).
- **class BertEncoder(nn.Module)**: المشفر الرئيسي - يكدّس 12 طبقة BertLayer.
- **class BertPooler(nn.Module)**: تجميع مخرجات [CLS] لتصنيف الجمل.
- **class BertPredictionHeadTransform(nn.Module)**: تحويل رأس التنبؤ (LayerNorm + GELU + Linear).
- **class BertLMPredictionHead(nn.Module)**: رأس تنبؤ الكلمات (Linear إلى حجم المفردات).
- **class BertOnlyMLMHead(nn.Module)**: رأس MLM لـ BERT.
- **class BertPreTrainedModel(PreTrainedModel)**: النموذج الأساسي قبل التدريب.
- **class BertModel(BertPreTrainedModel)**: النموذج الرئيسي (Encoder + Pooler).
- **class BertLMHeadModel(BertPreTrainedModel)**: نموذج اللغة مع رأس LM (للـ Decoder).

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: transformers (ACT2FN, ModelOutput, PreTrainedModel, BertConfig)
- **يُستدعى بواسطة**: extras/BLIP/models/blip.py, extras/BLIP/models/blip_itm.py, extras/BLIP/models/blip_nlvr.py, extras/BLIP/models/blip_pretrain.py, extras/BLIP/models/blip_retrieval.py, extras/BLIP/models/blip_vqa.py
- **الأثر البرمجي**: أساسي لجميع نماذج BLIP في المشروع
