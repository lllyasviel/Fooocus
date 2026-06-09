---
### 📄 docs/experiments_expansion.md

#### 1. الوظيفة الأساسية الفعالة (Core Purpose):
سكريبت اختبار (8 أسطر) لمحرك توسيع البرومبت FooocusExpansion. يُنشئ كائن expansion من modules.expansion.FooocusExpansion، ويُجرب توسيع النص 'a handsome man' مع 64 بذرة مختلفة (0 إلى 63) لمعاينة مخرجات التوسيع المختلفة.

#### 2. الكلاسات والدوال الرئيسية (Classes & Functions):
- **expansion**: كائن من FooocusExpansion (مُستورد من modules.expansion).
- **expansion(text, seed=i)**: استدعاء FooocusExpansion مع نص "a handsome man" وبذور من 0 إلى 63.

#### 3. خريطة الاعتماديات والعلاقات (File Dependencies):
- **يستدعي**: modules/expansion.py (FooocusExpansion)
- **يُستدعى بواسطة**: لا يُستدعى - سكريبت اختبار مستقل
- **الأثر البرمجي**: لا يؤثر على التطبيق الرئيسي - يُستخدم للتجربة والتطوير فقط

**للمزيد من التفاصيل، راجع:** [EXTRAS.md](EXTRAS.md)
