---
### 📄 مسار الملف الافتراضي: docs/wildcards/overview.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
ملفات wildcard (wildcards) النصية المستخدمة في توسيع البرومبتات. كل ملف يحتوي على قائمة كلمات سطر بسطر. تُستخدم في البرومبتات بتنسيق `{wildcard_name}`.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
**ملفات Wildcard المتاحة:**

| الملف | عدد الأسطر | المحتوى |
|-------|-------------|---------|
| **nationality.txt** | 195 | جنسيات (Afghan → Zimbabwean) |
| **flower.txt** | 250 | أنواع زهور (Acacia → Zinnia elegans) |
| **extended-color.txt** | 147 | ألوان CSS (aliceblue → yellowgreen) |
| **color.txt** | 17 | ألوان أساسية (aqua, black, blue, etc.) |
| **color_flower.txt** | 1 | wildcard مركّب: `__color__ __flower__` |
| **artist.txt** | 2765 | أسماء فنانين (2000+ فنان من جميع العصور) |
| **animal.txt** | 100 | حيوانات (Alligator → Zebra) |

**نظام Wildcard:**
- `__wildcard_name__` في البرومبت يُوسع إلى عنصر عشوائي من الملف
- يدعم التكامل (BFS) حتى عمق 64
- color_flower.txt يُظهرWildcard composition (يجمع color + flower)
- `read_wildcards_in_order` يقرأ العناصر بالترتيب بدلاً من العشوائي

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يعتمد على**: لا يعتمد على ملفات أخرى
- **يُستخدم بواسطة**: modules/util.py (دالة apply_wildcards), modules/config.py (يحمّل أسماء الملفات)
- **الأثر البرمجي**: إضافة ملف wildcard جديد يجعله متاحاً فوراً في البرومبتات
