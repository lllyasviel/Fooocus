---
### 📄 مسار الملف الافتراضي: docs/presets/overview.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملفات القوالب (Presets) تُعرّف إعدادات مسبقة للنماذج والأداء والأنماط. كل قالب JSON يحتوي على: النموذج الأساسي، Refiner، LoRAs، CFG scale، Sampler، Scheduler، الأداء، الأنماط، نسب العرض والارتفاع، وعناوين URL للتحميل التلقائي.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
**القوالب المتاحة:**

| القالب | النموذج | الأداء | CFG | الأنماط | الميزة الرئيسية |
|--------|---------|--------|-----|---------|----------------|
| **default.json** | juggernautXL_v8 | Speed | 4.0 | V2+Enhance+Sharp | Offset LoRA (0.1), 1152x896 |
| **anime.json** | animaPencilXL_v500 | Speed | 6.0 | V2+SemiRealistic+Masterpiece | بدون LoRA إضافي, 896x1152 |
| **realistic.json** | realisticStockPhoto_v20 | Speed | 3.0 | V2+Photograph+Negative | Film Photography LoRA (0.25) |
| **sai.json** | sd_xl_base_1.0 | Speed | 7.0 | V2+Cinematic | SDXL أساسي + Refiner 0.75 |
| **pony_v6.json** | ponyDiffusionV6XL | Speed | 7.0 | Pony فقط | VAE مخصص, بدون Inpaint |
| **playground_v2.5.json** | playground-v2.5-1024px | Speed | 2.0 | V2 فقط | EDM scheduler |
| **lightning.json** | juggernautXL_v8 | Lightning | 4.0 | V2+Enhance+Sharp | توليد سريع |
| **lcm.json** | juggernautXL_v8 | Extreme Speed | 4.0 | V2+Enhance+Sharp | LCM سريع |

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: لا يعتمد على ملفات أخرى
- **يُستخدم بواسطة**: modules/config.py (يحمّلها ويطبّقها), webui.py (يعرضها في القائمة المنسدلة)
- **الأثر البرمجي**: تغيير أي قالب سيتغير الإعدادات الافتراضية لذلك الوضع
