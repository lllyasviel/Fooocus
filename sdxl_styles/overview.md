---
### 📄 مسار الملف الافتراضي: docs/sdxl_styles/overview.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملفات تعريف الأنماط (Styles) لـ SDXL. كل ملف JSON يحتوي على قائمة أنماط، كل نمط يُعرّف: الاسم، البرومبت الإيجابي، البرومبت السلبي. الأنماط تُطبق على البرومبت قبل التوليد.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
**ملفات الأنماط المتاحة:**

| الملف | عدد الأنماط | الوصف |
|-------|-------------|-------|
| **sdxl_styles_fooocus.json** | 8 | أنماط Fooocus الأساسية (Enhance, Sharp, Masterpiece, Photograph, Negative, Cinematic, Pony) |
| **sdxl_styles_sai.json** | 16 | أنماط Stability AI الرسمية (3D Model, Analog Film, Anime, Cinematic, Digital Art, etc.) |
| **sdxl_styles_diva.json** | 82 | أنماط حركات فنية (Abstract Expressionism, Baroque, Bauhaus, Cubism, Impressionism, Pop Art, Surrealism, etc.) |
| **sdxl_styles_marc_k3nt3l.json** | 55 | أنماط تقنيات فنية/طباعة (Chromolithography, Cyanotype, Ukiyo-e, Vintage Posters, etc.) |
| **sdxl_styles_mre.json** | 21 | أنماط مظلمة/تكاسلية (Dark Dream, Underground, Dark Cyberpunk, Manga, etc.) |
| **sdxl_styles_twri.json** | 99 | أنماط مُصنّفة: Ads (9), Artstyle (18), Futuristic (10), Game (13), Misc (17), Papercraft (9), Photo (9) |

**بنية النمط الواحد:**
```json
{
    "name": "Style Name",
    "prompt": "positive prompt text {prompt} more text",
    "negative_prompt": "negative prompt text"
}
```

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: لا يعتمد على ملفات أخرى
- **يُستخدم بواسطة**: modules/sdxl_styles.py (يحمّلها ويطبّقها), modules/config.py (يتحقق من الأسماء القانونية)
- **الأثر البرمجي**: إضافة نمط جديد هنا يجعله متاحاً فوراً في واجهة المستخدم
