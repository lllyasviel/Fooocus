<div align=center>
<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/483fb86d-c9a2-4c20-997c-46dafc124f25">
</div>

# فوكس (Fooocus)

[>>> انقر هنا لتثبيت فوكس <<<](#download)

فوكس هو برنامج لتوليد الصور (مبني على [Gradio](https://www.gradio.app/) <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>).

**للمزيد من التفاصيل، راجع الملفات التالية:**
- [pdf.md](pdf.md) - الدليل الشامل
- [map-road.md](map-road.md) - خريطة طريق Fooocus
- [troubleshoot_ar.md](troubleshoot_ar.md) - استكشاف الأخطاء
- [update_log_ar.md](update_log_ar.md) - سجل التحديثات

يقدم فوكس إعادة تفكير في تصميم مولدات الصور. البرنامج يعمل دون اتصال، مفتوح المصدر، ومجاني، وفي الوقت نفسه، مشابه للعديد من مولدات الصور عبر الإنترنت مثل ميدجورني، فلا حاجة للتعديل اليدوي، ويحتاج المستخدمون فقط إلى التركيز على التوجيهات (prompts) والصور. كما قام فوكس بتبسيط التثبيت: بين الضغط على "تنزيل" وتوليد أول صورة، عدد النقرات المطلوبة بالماوس محدود بدقة بأقل من 3. الحد الأدنى المطلوب لذاكرة GPU هو 4 جيجابايت (Nvidia).

**مؤخرًا، توجد العديد من المواقع المزيفة على جوجل عند البحث عن "fooocus". لا تثق بتلك المواقع – هنا المصدر الرسمي الوحيد لفوكس.**

# حالة المشروع: دعم طويل الأمد محدود (LTS) مع إصلاحات الأخطاء فقط

مشروع فوكس، المبني بالكامل على بنية **Stable Diffusion XL**، هو الآن في حالة دعم طويل الأمد محدود (LTS) مع إصلاحات الأخطاء فقط. نظرًا لأن الوظائف الحالية تُعتبر خالية تقريبًا من المشكلات البرمجية (بفضل جهود [mashb1t](https://github.com/mashb1t) الكبيرة)، ستركز التحديثات المستقبلية حصريًا على معالجة أي أخطاء قد تطرأ.

**لا توجد خطط حالية للانتقال إلى أو دمج بنى نماذج أحدث.** ومع ذلك، قد يتغير هذا بمرور الوقت مع تطور مجتمع المصادر المفتوحة. على سبيل المثال، إذا تقارب المجتمع حول طريقة واحدة مهيمنة لتوليد الصور (وهو ما قد يحدث بالفعل في غضون نصف سنة أو سنة بالنظر إلى الوضع الحالي)، فقد ينتقل فوكس أيضًا إلى تلك الطريقة بالضبط.

بالنسبة للمهتمين باستخدام نماذج أحدث مثل **Flux**، نوصي باستكشاف منصات بديلة مثل [WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) (أيضًا منا)، [ComfyUI/SwarmUI](https://github.com/comfyanonymous/ComfyUI). بالإضافة إلى ذلك، تتوفر العديد من [التفرعات الممتازة لفوكس](https://github.com/lllyasviel/Fooocus?tab=readme-ov-file#forks) للتجربة.

مرة أخرى، مؤخرًا توجد العديد من المواقع المزيفة على جوجل عند البحث عن "fooocus". لا **تحصل** على فوكس من تلك المواقع – هذه الصفحة هي المصدر الرسمي الوحيد لفوكس. ليس لدينا أي موقع مثل "fooocus.com" أو "fooocus.net" أو "fooocus.co" أو "fooocus.ai" أو "fooocus.org" أو "fooocus.pro" أو "fooocus.one". تلك المواقع كلها **مزيفة**. **ليست لها أي علاقة بنا على الإطلاق. فوكس هو برنامج مفتوح المصدر غير تجاري يعمل دون اتصال بنسبة 100%.**

# الميزات

فيما يلي قائمة سريعة باستخدام أمثلة ميدجورني:

| ميدجورني | فوكس |
| - | - |
| تحويل نص إلى صورة عالي الجودة دون الحاجة لكثير من هندسة التوجيه أو ضبط المعايير. <br> (طريقة غير معروفة) | تحويل نص إلى صورة عالي الجودة دون الحاجة لكثير من هندسة التوجيه أو ضبط المعايير. <br> (يمتلك فوكس محرك معالجة توجيه غير متصل يعتمد على GPT-2 والعديد من تحسينات أخذ العينات بحيث تكون النتائج دائمًا جميلة، بغض النظر عما إذا كان توجيهك قصيرًا مثل "منزل في حديقة" أو طويلاً يصل إلى 1000 كلمة) |
| V1 V2 V3 V4 | صورة إدخال -> تحسين أو تباين -> تغيير (خفيف) / تغيير (قوي) |
| U1 U2 U3 U4 | صورة إدخال -> تحسين أو تباين -> تحسين (1.5x) / تحسين (2x) |
| الرسم الداخلي / أعلى / أسفل / يسار / يمين (تحريك) | صورة إدخال -> رسم داخلي أو رسم خارجي -> رسم داخلي / أعلى / أسفل / يسار / يمين <br> (يستخدم فوكس خوارزمية الرسم الداخلي الخاصة به ونماذج الرسم الداخلي بحيث تكون النتائج أكثر إرضاءً من جميع البرامج الأخرى التي تستخدم طريقة/نموذج الرسم الداخلي القياسي لـ SDXL) |
| توجيه الصورة | صورة إدخال -> توجيه الصورة <br> (يستخدم فوكس خوارزمية توجيه الصور الخاصة به بحيث تكون جودة النتيجة وفهم التوجيه أكثر إرضاءً من جميع البرامج الأخرى التي تستخدم طرق SDXL القياسية مثل IP-Adapters القياسية أو Revisions) |
| --style | متقدم -> النمط |
| --stylize | متقدم -> متقدم -> التوجيه |
| --niji | [مشغلات متعددة: "run.bat" و "run_anime.bat" و "run_realistic.bat".](https://github.com/lllyasviel/Fooocus/discussions/679) <br> يدعم فوكس نماذج SDXL على Civitai <br> (يمكنك البحث في جوجل عن "Civitai" إذا كنت لا تعرفه) |
| --quality | متقدم -> الجودة |
| --repeat | متقدم -> عدد الصور |
| توجيهات متعددة (::) | فقط استخدم عدة أسطر من التوجيهات |
| أوزان التوجيه | يمكنك استخدام "أنا (سعيد:1.5)". <br> يستخدم فوكس خوارزمية إعادة الوزن لـ A1111 بحيث تكون النتائج أفضل من ComfyUI إذا قام المستخدمون بنسخ التوجيهات مباشرة من Civitai. (لأنه إذا كانت التوجيهات مكتوبة بإعادة وزن ComfyUI، فمن غير المرجح أن ينسخ المستخدمون نصوص التوجيهات حيث يفضلون سحب الملفات) <br> لاستخدام التضمين (embedding)، يمكنك استخدام "(embedding:file_name:1.1)" |
| --no | متقدم -> التوجيه السلبي |
| --ar | متقدم -> نسب الأبعاد |
| InsightFace | صورة إدخال -> توجيه الصورة -> متقدم -> تبديل الوجوه |
| وصف | صورة إدخال -> وصف |

فيما يلي قائمة سريعة باستخدام أمثلة LeonardoAI:

| LeonardoAI | فوكس |
| - | - |
| سحر التوجيه | متقدم -> النمط -> Fooocus V2 |
| معايير أخذ العينات المتقدمة (مثل التباين/الحدّة/إلخ) | متقدم -> متقدم -> حدّة أخذ العينات / إلخ |
| شبكات التحكم سهلة الاستخدام | صورة إدخال -> توجيه الصورة -> متقدم |

أيضًا، [انقر هنا لتصفح الميزات المتقدمة.](https://github.com/lllyasviel/Fooocus/discussions/117)

# التنزيل

### ويندوز

يمكنك تنزيل فوكس مباشرة من:

**[>>> انقر هنا للتنزيل <<<](https://github.com/lllyasviel/Fooocus/releases/download/v2.5.0/Fooocus_win64_2-5-0.7z)**

بعد تنزيل الملف، يرجى فك ضغطه ثم تشغيل "run.bat".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

عند تشغيل البرنامج لأول مرة، سيقوم تلقائيًا بتنزيل النماذج:

1. سيقوم بتنزيل [النماذج الافتراضية](#models) إلى المجلد "Fooocus\models\checkpoints" وفقًا للإعدادات المسبقة المختلفة. يمكنك تنزيلها مسبقًا إذا كنت لا تريد التنزيل التلقائي.
2. لاحظ أنه إذا كنت تستخدم الرسم الداخلي (inpaint)، في المرة الأولى التي ترسم فيها صورة داخليًا، سيقوم بتنزيل [نموذج التحكم في الرسم الداخلي الخاص بفوكس من هنا](https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch) كملف "Fooocus\models\inpaint\inpaint_v26.fooocus.patch" (حجم هذا الملف 1.28 جيجابايت).

بعد فوكس 2.1.60، سيكون لديك أيضًا `run_anime.bat` و `run_realistic.bat`. هما إعدادات مسبقة مختلفة للنماذج (وتتطلب نماذج مختلفة، ولكن سيتم تنزيلها تلقائيًا). [تحقق هنا لمزيد من التفاصيل](https://github.com/lllyasviel/Fooocus/discussions/679).

بعد فوكس 2.3.0، يمكنك أيضًا تبديل الإعدادات المسبقة مباشرة في المتصفح. ضع في اعتبارك إضافة هذه الوسائط إذا كنت تريد تغيير السلوك الافتراضي:
* استخدم `--disable-preset-selection` لتعطيل اختيار الإعداد المسبق في المتصفح.
* استخدم `--always-download-new-model` لتنزيل النماذج المفقودة عند تبديل الإعداد المسبق. الوضع الافتراضي هو الرجوع إلى `previous_default_models` المحدد في الإعداد المسبق المقابل، انظر أيضًا مخرجات الطرفية.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

إذا كان لديك هذه الملفات بالفعل، يمكنك نسخها إلى المواقع أعلاه لتسريع التثبيت.

لاحظ أنه إذا رأيت **"MetadataIncompleteBuffer" أو "PytorchStreamReader"**، فإن ملفات النموذج لديك تالفة. يرجى تنزيل النماذج مرة أخرى.

فيما يلي اختبار على كمبيوتر محمول منخفض المواصفات نسبيًا مع **16 جيجابايت ذاكرة نظام (RAM)** و **6 جيجابايت ذاكرة فيديو (VRAM)** (كمبيوتر محمول Nvidia 3060). السرعة على هذا الجهاز حوالي 1.35 ثانية لكل تكرار. مثير للإعجاب – أجهزة الكمبيوتر المحمولة المزودة بـ 3060 عادة ما تكون بسعر مقبول جدًا هذه الأيام.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

علاوة على ذلك، أفادت العديد من البرامج الأخرى مؤخرًا أن برنامج تشغيل Nvidia أعلى من 532 يكون أحيانًا أبطأ 10 مرات من برنامج تشغيل Nvidia 531. إذا كان وقت التوليد طويلاً جدًا، فكر في تنزيل [Nvidia Driver 531 Laptop](https://www.nvidia.com/download/driverResults.aspx/199991/en-us/) أو [Nvidia Driver 531 Desktop](https://www.nvidia.com/download/driverResults.aspx/199990/en-us/).

لاحظ أن الحد الأدنى المطلوب هو **4 جيجابايت ذاكرة GPU Nvidia (4GB VRAM)** و **8 جيجابايت ذاكرة نظام (8GB RAM)**. يتطلب هذا استخدام تقنية المبادلة الافتراضية من مايكروسوفت (Virtual Swap)، والتي يتم تمكينها تلقائيًا بواسطة تثبيت ويندوز في معظم الحالات، لذلك غالبًا لا تحتاج لفعل أي شيء حيال ذلك. ومع ذلك، إذا لم تكن متأكدًا، أو إذا قمت بتعطيلها يدويًا (هل سيفعل أحد ذلك حقًا؟)، أو **إذا رأيت أي "RuntimeError: CPUAllocator"**، يمكنك تمكينها هنا:

<details>
<summary>انقر هنا لرؤية تعليمات الصورة. </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

**وتأكد من أن لديك على الأقل 40 جيجابايت مساحة خالية على كل محرك أقراص إذا كنت لا تزال ترى "RuntimeError: CPUAllocator" !**

</details>

يرجى فتح مشكلة (issue) إذا كنت تستخدم أجهزة مماثلة ولكنك لا تزال غير قادر على تحقيق أداء مقبول.

لاحظ أن [الحد الأدنى المطلوب](#minimal-requirement) يختلف باختلاف المنصات.

انظر أيضًا المشكلات الشائعة واستكشاف الأخطاء وإصلاحها [هنا](troubleshoot.md).

### كولاب (Colab)

(آخر اختبار - 12 أغسطس 2024 بواسطة [mashb1t](https://github.com/mashb1t))

| كولاب | معلومات |
| --- | --- |
[![فتح في كولاب](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/fooocus_colab.ipynb) | فوكس الرسمي

في كولاب، يمكنك تعديل السطر الأخير إلى `!python entry_with_update.py --share --always-high-vram` أو `!python entry_with_update.py --share --always-high-vram --preset anime` أو `!python entry_with_update.py --share --always-high-vram --preset realistic` لإصدارات فوكس الافتراضي/الأنمي/الواقعي.

يمكنك أيضًا تغيير الإعداد المسبق في واجهة المستخدم. يرجى العلم أن هذا قد يؤدي إلى انتهاء المهلة بعد 60 ثانية. إذا كان الأمر كذلك، يرجى الانتظار حتى يكتمل التنزيل، وقم بتغيير الإعداد المسبق إلى الإعداد الأولي ثم العودة إلى الإعداد الذي اخترته أو أعد تحميل الصفحة.

لاحظ أن هذا الكولاب سيعطل المُحسِّن (refiner) افتراضيًا لأن موارد كولاب المجانية محدودة نسبيًا (وبعض الميزات "الكبيرة" مثل توجيه الصورة قد تتسبب في قطع اتصال كولاب المجاني). نحن نضمن أن تحويل النص إلى صورة الأساسي يعمل دائمًا على كولاب المجاني.

استخدام `--always-high-vram` ينقل تخصيص الموارد من RAM إلى VRAM ويحقق أفضل توازن عام بين الأداء والمرونة والاستقرار على مثيل T4 الافتراضي. يرجى العثور على مزيد من المعلومات [هنا](https://github.com/lllyasviel/Fooocus/pull/1710#issuecomment-1989185346).

شكرًا لـ [camenduru](https://github.com/camenduru) على القالب!

### لينكس (باستخدام Anaconda)

إذا كنت تريد استخدام Anaconda/Miniconda، يمكنك

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

ثم قم بتنزيل النماذج: قم بتنزيل [النماذج الافتراضية](#models) إلى المجلد "Fooocus\models\checkpoints". **أو دع فوكس يقوم بتنزيل النماذج تلقائيًا** باستخدام المُشغل:

    conda activate fooocus
    python entry_with_update.py

أو، إذا كنت تريد فتح منفذ بعيد، استخدم

    conda activate fooocus
    python entry_with_update.py --listen

استخدم `python entry_with_update.py --preset anime` أو `python entry_with_update.py --preset realistic` لإصدار فوكس للأنمي/الواقعي.

### لينكس (باستخدام Python Venv)

يجب أن يكون لديك **Python 3.10** مثبتًا على لينكس، ولنفترض أنه يمكن استدعاء Python الخاص بك بالأمر **python3** مع عمل نظام venv الخاص بك؛ يمكنك

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    python3 -m venv fooocus_env
    source fooocus_env/bin/activate
    pip install -r requirements_versions.txt

انظر الأقسام أعلاه لتنزيل النماذج. يمكنك تشغيل البرنامج بـ:

    source fooocus_env/bin/activate
    python entry_with_update.py

أو، إذا كنت تريد فتح منفذ بعيد، استخدم

    source fooocus_env/bin/activate
    python entry_with_update.py --listen

استخدم `python entry_with_update.py --preset anime` أو `python entry_with_update.py --preset realistic` لإصدار فوكس للأنمي/الواقعي.

### لينكس (باستخدام Python النظام الأصلي)

إذا كنت تعرف ما تفعله، وكان لينكس لديك مثبتًا عليه **Python 3.10** بالفعل، ويمكن استدعاء Python الخاص بك بالأمر **python3** (و Pip بـ **pip3**)، يمكنك

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    pip3 install -r requirements_versions.txt

انظر الأقسام أعلاه لتنزيل النماذج. يمكنك تشغيل البرنامج بـ:

    python3 entry_with_update.py

أو، إذا كنت تريد فتح منفذ بعيد، استخدم

    python3 entry_with_update.py --listen

استخدم `python entry_with_update.py --preset anime` أو `python entry_with_update.py --preset realistic` لإصدار فوكس للأنمي/الواقعي.

### لينكس (بطاقات AMD GPU)

لاحظ أن [الحد الأدنى المطلوب](#minimal-requirement) يختلف باختلاف المنصات.

نفس التعليمات أعلاه. تحتاج إلى تغيير torch إلى إصدار AMD

    pip uninstall torch torchvision torchaudio torchtext functorch xformers 
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

ومع ذلك، لم يتم اختبار AMD بشكل مكثف. دعم AMD في مرحلة تجريبية (beta).

استخدم `python entry_with_update.py --preset anime` أو `python entry_with_update.py --preset realistic` لإصدار فوكس للأنمي/الواقعي.

### ويندوز (بطاقات AMD GPU)

لاحظ أن [الحد الأدنى المطلوب](#minimal-requirement) يختلف باختلاف المنصات.

نفس نظام ويندوز. قم بتنزيل البرنامج وتعديل محتوى `run.bat` كما يلي:

    .\python_embeded\python.exe -m pip uninstall torch torchvision torchaudio torchtext functorch xformers -y
    .\python_embeded\python.exe -m pip install torch-directml
    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --directml
    pause

ثم قم بتشغيل `run.bat`.

ومع ذلك، لم يتم اختبار AMD بشكل مكثف. دعم AMD في مرحلة تجريبية (beta).

لـ AMD، استخدم `.\python_embeded\python.exe Fooocus\entry_with_update.py --directml --preset anime` أو `.\python_embeded\python.exe Fooocus\entry_with_update.py --directml --preset realistic` لإصدار فوكس للأنمي/الواقعي.

### ماك

لاحظ أن [الحد الأدنى المطلوب](#minimal-requirement) يختلف باختلاف المنصات.

لم يتم اختبار ماك بشكل مكثف. فيما يلي إرشادات غير رسمية لاستخدام ماك. يمكنك مناقشة المشكلات [هنا](https://github.com/lllyasviel/Fooocus/pull/129).

يمكنك تثبيت فوكس على Apple Mac silicon (M1 أو M2) مع macOS 'Catalina' أو إصدار أحدث. يعمل فوكس على أجهزة Apple silicon عبر [PyTorch](https://pytorch.org/get-started/locally/) تسريع جهاز MPS. أجهزة Mac Silicon لا تأتي مع بطاقة رسوميات مخصصة، مما يؤدي إلى أوقات معالجة صور أطول بكثير مقارنة بأجهزة الكمبيوتر المزودة ببطاقات رسوميات مخصصة.

1. قم بتثبيت مدير حزم conda و pytorch nightly. اقرأ دليل مطوري Apple [تدريب PyTorch المتسارع على Mac](https://developer.apple.com/metal/pytorch/) للحصول على التعليمات. تأكد من أن pytorch يتعرف على جهاز MPS الخاص بك.
1. افتح تطبيق Terminal في macOS واستنسخ هذا المستودع بـ `git clone https://github.com/lllyasviel/Fooocus.git`.
1. انتقل إلى دليل فوكس الجديد، `cd Fooocus`.
1. أنشئ بيئة conda جديدة، `conda env create -f environment.yaml`.
1. قم بتفعيل بيئة conda الجديدة الخاصة بك، `conda activate fooocus`.
1. قم بتثبيت الحزم المطلوبة بواسطة فوكس، `pip install -r requirements_versions.txt`.
1. شغّل فوكس بتشغيل `python entry_with_update.py`. (قد يحتاج بعض مستخدمي Mac M2 إلى `python entry_with_update.py --disable-offload-from-vram` لتسريع تحميل/تفريغ النماذج.) في المرة الأولى التي تشغل فيها فوكس، سيقوم تلقائيًا بتنزيل نماذج Stable Diffusion SDXL وسيستغرق وقتًا طويلاً، اعتمادًا على اتصالك بالإنترنت.

استخدم `python entry_with_update.py --preset anime` أو `python entry_with_update.py --preset realistic` لإصدار فوكس للأنمي/الواقعي.

### دوكر (Docker)

انظر [docker.md](docker.md)

### تنزيل الإصدار السابق

انظر الإرشادات [هنا](https://github.com/lllyasviel/Fooocus/discussions/1405).

## الحد الأدنى من المتطلبات

فيما يلي الحد الأدنى من المتطلبات لتشغيل فوكس محليًا. إذا كانت قدرات جهازك أقل من هذه المواصفات، فقد لا تتمكن من استخدام فوكس محليًا. (يرجى إعلامنا، في أي حال، إذا كانت قدرات جهازك أقل ولكن فوكس لا يزال يعمل.)

| نظام التشغيل     | GPU                          | الحد الأدنى لذاكرة GPU           | الحد الأدنى لذاكرة النظام     | [مبادلة النظام](troubleshoot.md) | ملاحظة                                                                      |
|-------------------|------------------------------|------------------------------|---------------------------|--------------------------------|-----------------------------------------------------------------------------|
| ويندوز/لينكس      | Nvidia RTX 4XXX              | 4GB                          | 8GB                       | مطلوب                         | الأسرع                                                                      |
| ويندوز/لينكس      | Nvidia RTX 3XXX              | 4GB                          | 8GB                       | مطلوب                         | عادة أسرع من RTX 2XXX                                                        |
| ويندوز/لينكس      | Nvidia RTX 2XXX              | 4GB                          | 8GB                       | مطلوب                         | عادة أسرع من GTX 1XXX                                                        |
| ويندوز/لينكس      | Nvidia GTX 1XXX              | 8GB (&ast; 6GB غير مؤكد)     | 8GB                       | مطلوب                         | أسرع بشكل هامشي فقط من CPU                                                   |
| ويندوز/لينكس      | Nvidia GTX 9XX               | 8GB                          | 8GB                       | مطلوب                         | أسرع أو أبطأ من CPU                                                         |
| ويندوز/لينكس      | Nvidia GTX < 9XX             | غير مدعوم                    | /                         | /                              | /                                                                           |
| ويندوز            | AMD GPU                      | 8GB    (آخر تحديث 30 ديسمبر 2023) | 8GB                       | مطلوب                         | عبر DirectML (&ast; ROCm معلق)، أبطأ بحوالي 3 مرات من Nvidia RTX 3XXX        |
| لينكس             | AMD GPU                      | 8GB                          | 8GB                       | مطلوب                         | عبر ROCm، أبطأ بحوالي 1.5 مرة من Nvidia RTX 3XXX                            |
| ماك               | M1/M2 MPS                    | مشترك                       | مشترك                    | مشترك                         | أبطأ بحوالي 9 مرات من Nvidia RTX 3XXX                                        |
| ويندوز/لينكس/ماك  | استخدام CPU فقط              | 0GB                          | 32GB                      | مطلوب                         | أبطأ بحوالي 17 مرة من Nvidia RTX 3XXX                                       |

&ast; AMD GPU ROCm (معلق): لا تزال AMD تعمل على دعم ROCm على ويندوز.

&ast; Nvidia GTX 1XXX 6GB غير مؤكد: يبلغ بعض الأشخاص عن نجاح 6GB على GTX 10XX، ولكن يبلغ آخرون عن حالات فشل.

*لاحظ أن فوكس مخصص فقط لتوليد الصور عالية الجودة للغاية. لن ندعم النماذج الأصغر لتقليل المتطلبات والتضحية بجودة النتيجة.*

## استكشاف الأخطاء وإصلاحها

انظر المشكلات الشائعة [هنا](troubleshoot.md).

## النماذج الافتراضية
<a name="models"></a>

بالنظر إلى الأهداف المختلفة، تختلف النماذج والإعدادات الافتراضية لفوكس:

| المهمة     | ويندوز | وسائط لينكس | النموذج الرئيسي              | المُحسِّن (Refiner) | الإعدادات                                                                     |
|-----------| --- | --- |-----------------------------| --- |--------------------------------------------------------------------------------|
| عام       | run.bat |  | juggernautXL_v8Rundiffusion | غير مستخدم | [هنا](https://github.com/lllyasviel/Fooocus/blob/main/presets/default.json)   |
| واقعي     | run_realistic.bat | --preset realistic | realisticStockPhoto_v20     | غير مستخدم | [هنا](https://github.com/lllyasviel/Fooocus/blob/main/presets/realistic.json) |
| أنمي      | run_anime.bat | --preset anime | animaPencilXL_v500          | غير مستخدم | [هنا](https://github.com/lllyasviel/Fooocus/blob/main/presets/anime.json)     |

لاحظ أن التنزيل **تلقائي** - لا تحتاج لفعل أي شيء إذا كان اتصال الإنترنت جيدًا. ومع ذلك، يمكنك تنزيلها يدويًا إذا كان لديك (أو نقلها من مكان آخر) تحضيرك الخاص.

## الوصول إلى واجهة المستخدم والمصادقة

بالإضافة إلى التشغيل على localhost، يمكن لفوكس أيضًا عرض واجهة المستخدم الخاصة به بطريقتين:
* مستمع واجهة المستخدم المحلي: استخدم `--listen` (حدد المنفذ مثلاً بـ `--port 8888`).
* الوصول عبر API: استخدم `--share` (يسجل نقطة نهاية في `.gradio.live`).

في كلتا الطريقتين، الوصول غير مصادق عليه افتراضيًا. يمكنك إضافة مصادقة أساسية عن طريق إنشاء ملف يسمى `auth.json` في الدليل الرئيسي، والذي يحتوي على قائمة من كائنات JSON بمفاتيح `user` و `pass` (انظر المثال في [auth-example.json](./auth-example.json)).

## قائمة الحيل "المخفية"
<a name="tech_list"></a>

<details>
<summary>انقر لرؤية قائمة الحيل. هذه مبنية على SDXL وليست محدثة جدًا مع أحدث النماذج.</summary>

1. [توسيع التوجيه المبني على GPT-2 كنمط ديناميكي "Fooocus V2".](https://github.com/lllyasviel/Fooocus/discussions/117#raw) (مشابه للمعالجة المسبقة المخفية ووضع "raw" في ميدجورني، أو سحر التوجيه في LeonardoAI).
2. التبديل الأصلي للمُحسِّن داخل k-sampler واحد. الميزة هي أن نموذج المُحسِّن يمكنه الآن إعادة استخدام زخم النموذج الأساسي (أو معلمات تاريخ ODE) التي تم جمعها من أخذ العينات k لتحقيق أخذ عينات أكثر تماسكًا. في الإصلاح عالي الدقة لـ Automatic1111 ونظام العقد في ComfyUI، يستخدم النموذج الأساسي والمُحسِّن اثنين من k-samplers مستقلين، مما يعني أن الزخم يُهدر إلى حد كبير، وتنقطع استمرارية أخذ العينات. يستخدم فوكس أخذ العينات k-diffusion المتقدم الخاص به الذي يضمن تبديلًا سلسًا وأصليًا ومستمرًا في إعداد المُحسِّن. (تحديث 13 أغسطس: في الواقع، ناقشت هذا مع Automatic1111 قبل عدة أيام، ويبدو أن "التبديل الأصلي للمُحسِّن داخل k-sampler واحد" قد تم [دمجه](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12371) في الفرع التطويري لـ webui. عظيم!)
3. توجيه ADM السلبي. نظرًا لأن أعلى مستوى دقة في XL Base لا يحتوي على انتباه متبادل (cross attentions)، فإن الإشارات الإيجابية والسلبية لأعلى مستوى دقة في XL لا يمكنها تلقي تباين كافٍ أثناء أخذ عينات CFG، مما يتسبب في ظهور النتائج بلاستيكية بعض الشيء أو ناعمة بشكل مفرط في حالات معينة. لحسن الحظ، نظرًا لأن أعلى مستوى دقة في XL لا يزال مشروطًا بنسب أبعاد الصورة (ADM)، يمكننا تعديل ADM على الجانب الإيجابي/السلبي للتعويض عن نقص تباين CFG في أعلى مستوى دقة. (تحديث 16 أغسطس، تطبيق iOS [Draw Things](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820) سيدعم توجيه ADM السلبي. عظيم!)
4. قمنا بتنفيذ تباين مضبوط بعناية للقسم 5.1 من ["تحسين جودة عينة نماذج الانتشار باستخدام توجيه الانتباه الذاتي"](https://arxiv.org/pdf/2210.00939.pdf). تم ضبط الوزن على مستوى منخفض جدًا، ولكن هذا هو الضمان النهائي لفوكس للتأكد من أن XL لن ينتج أبدًا مظهرًا ناعمًا أو بلاستيكيًا مفرطًا (أمثلة [هنا](https://github.com/lllyasviel/Fooocus/discussions/117#sharpness)). يمكن لهذا تقريبًا القضاء على جميع الحالات التي لا يزال XL ينتج فيها نتائج ناعمة بشكل مفرط، حتى مع توجيه ADM السلبي. (تحديث 18 أغسطس 2023، تم تغيير نواة Gaussian لـ SAG إلى نواة متباينة الخواص للحفاظ على بنية أفضل وتقليل القطع الأثرية.)
5. قمنا بتعديل قوالب الأنماط قليلاً وأضفنا "cinematic-default".
6. اختبرنا "sd_xl_offset_example-lora_1.0.safetensors" ويبدو أنه عندما يكون وزن lora أقل من 0.5، تكون النتائج دائمًا أفضل من XL بدون lora.
7. معلمات آلات أخذ العينات مضبوطة بعناية.
8. نظرًا لأن XL يستخدم الترميز الموضعي (positional encoding) لدقة التوليد، فإن الصور المولدة بعدة دقات ثابتة تبدو أفضل قليلاً من تلك ذات الدقات العشوائية (لأن الترميز الموضعي ليس جيدًا جدًا في معالجة الأعداد الصحيحة غير المرئية أثناء التدريب). يشير هذا إلى أن الدقات في واجهة المستخدم قد تكون مشفرة بشكل ثابت للحصول على أفضل النتائج.
9. يبدو أن التوجيهات المنفصلة لمشفرَي نص مختلفين غير ضرورية. قد تعمل التوجيهات المنفصلة للنموذج الأساسي والمُحسِّن، لكن التأثيرات عشوائية، ونمتنع عن تنفيذ هذا.
10. تبدو عائلة DPM مناسبة لـ XL لأن XL يولد أحيانًا نسيجًا ناعمًا بشكل مفرط، ولكن عائلة DPM تولد أحيانًا تفاصيل كثيفة بشكل مفرط في النسيج. تأثيرهما المشترك يبدو محايدًا وجذابًا للإدراك البشري.
11. نظام مصمم بعناية لموازنة أنماط متعددة بالإضافة إلى توسيع التوجيه.
12. استخدام طريقة automatic1111 لتطبيع توكيد التوجيه. هذا يحسن النتائج بشكل ملحوظ عندما ينسخ المستخدمون التوجيهات مباشرة من civitai.
13. نظام التبديل المشترك للمُحسِّن يدعم الآن أيضًا img2img والتحسين (upscale) بطريقة سلسة.
14. مقياس CFG وتصحيح TSNR (مضبوط لـ SDXL) عندما يكون CFG أكبر من 10.
</details>

## التخصيص

بعد أول مرة تقوم فيها بتشغيل فوكس، سيتم إنشاء ملف إعدادات في `Fooocus\config.txt`. يمكن تحرير هذا الملف لتغيير مسار النموذج أو المعايير الافتراضية.

على سبيل المثال، قد يبدو ملف `Fooocus\config.txt` مُحرر (سيتم إنشاء هذا الملف بعد التشغيل الأول) بهذا الشكل:

```json
{
    "path_checkpoints": "D:\\Fooocus\\models\\checkpoints",
    "path_loras": "D:\\Fooocus\\models\\loras",
    "path_embeddings": "D:\\Fooocus\\models\\embeddings",
    "path_vae_approx": "D:\\Fooocus\\models\\vae_approx",
    "path_upscale_models": "D:\\Fooocus\\models\\upscale_models",
    "path_inpaint": "D:\\Fooocus\\models\\inpaint",
    "path_controlnet": "D:\\Fooocus\\models\\controlnet",
    "path_clip_vision": "D:\\Fooocus\\models\\clip_vision",
    "path_fooocus_expansion": "D:\\Fooocus\\models\\prompt_expansion\\fooocus_expansion",
    "path_outputs": "D:\\Fooocus\\outputs",
    "default_model": "realisticStockPhoto_v10.safetensors",
    "default_refiner": "",
    "default_loras": [["lora_filename_1.safetensors", 0.5], ["lora_filename_2.safetensors", 0.5]],
    "default_cfg_scale": 3.0,
    "default_sampler": "dpmpp_2m",
    "default_scheduler": "karras",
    "default_negative_prompt": "low quality",
    "default_positive_prompt": "",
    "default_styles": [
        "Fooocus V2",
        "Fooocus Photograph",
        "Fooocus Negative"
    ]
}
```

العديد من المفاتيح والتنسيقات والأمثلة الأخرى موجودة في `Fooocus\config_modification_tutorial.txt` (سيتم إنشاء هذا الملف بعد التشغيل الأول).

فكر مرتين قبل أن تقوم فعليًا بتغيير الإعدادات. إذا وجدت أنك تسبب في كسر الأشياء، فقط قم بحذف `Fooocus\config.txt`. سيعود فوكس إلى الإعدادات الافتراضية.

طريقة أكثر أمانًا هي مجرد تجربة "run_anime.bat" أو "run_realistic.bat" - يجب أن تكونا جيدتين بما يكفي لمهام مختلفة.

~لاحظ أن `user_path_config.txt` مهمل وسيتم إزالته قريبًا.~ (تحرير: تمت إزالته بالفعل.)

### جميع أوامر CMD

```
entry_with_update.py  [-h] [--listen [IP]] [--port PORT]
                      [--disable-header-check [ORIGIN]]
                      [--web-upload-size WEB_UPLOAD_SIZE]
                      [--hf-mirror HF_MIRROR]
                      [--external-working-path PATH [PATH ...]]
                      [--output-path OUTPUT_PATH]
                      [--temp-path TEMP_PATH] [--cache-path CACHE_PATH]
                      [--in-browser] [--disable-in-browser]
                      [--gpu-device-id DEVICE_ID]
                      [--async-cuda-allocation | --disable-async-cuda-allocation]
                      [--disable-attention-upcast]
                      [--all-in-fp32 | --all-in-fp16]
                      [--unet-in-bf16 | --unet-in-fp16 | --unet-in-fp8-e4m3fn | --unet-in-fp8-e5m2]
                      [--vae-in-fp16 | --vae-in-fp32 | --vae-in-bf16]
                      [--vae-in-cpu]
                      [--clip-in-fp8-e4m3fn | --clip-in-fp8-e5m2 | --clip-in-fp16 | --clip-in-fp32]
                      [--directml [DIRECTML_DEVICE]]
                      [--disable-ipex-hijack]
                      [--preview-option [none,auto,fast,taesd]]
                      [--attention-split | --attention-quad | --attention-pytorch]
                      [--disable-xformers]
                      [--always-gpu | --always-high-vram | --always-normal-vram | --always-low-vram | --always-no-vram | --always-cpu [CPU_NUM_THREADS]]
                      [--always-offload-from-vram]
                      [--pytorch-deterministic] [--disable-server-log]
                      [--debug-mode] [--is-windows-embedded-python]
                      [--disable-server-info] [--multi-user] [--share]
                      [--preset PRESET] [--disable-preset-selection]
                      [--language LANGUAGE]
                      [--disable-offload-from-vram] [--theme THEME]
                      [--disable-image-log] [--disable-analytics]
                      [--disable-metadata] [--disable-preset-download]
                      [--disable-enhance-output-sorting]
                      [--enable-auto-describe-image]
                      [--always-download-new-model]
                      [--rebuild-hash-cache [CPU_NUM_THREADS]]
```

## ميزات التوجيه المضمنة

### البطاقات الجامحة (Wildcards)

مثال توجيه: `__color__ flower`

تتم معالجته للتوجيه الإيجابي والسلبي.

يختار بطاقة جامحة عشوائية من قائمة خيارات محددة مسبقًا، في هذه الحالة ملف `wildcards/color.txt`.
سيتم استبدال البطاقة الجامحة بلون عشوائي (العشوائية تعتمد على البذرة (seed)).
يمكنك أيضًا تعطيل العشوائية ومعالجة ملف البطاقات الجامحة من الأعلى إلى الأسفل عن طريق تفعيل خانة الاختيار `Read wildcards in order` في وضع تصحيح المطور.

يمكن تداخل البطاقات الجامحة ودمجها، ويمكن استخدام بطاقات جامحة متعددة في نفس التوجيه (مثال انظر `wildcards/color_flower.txt`).

### معالجة المصفوفات

مثال توجيه: `[[red, green, blue]] flower`

تتم معالجته فقط للتوجيه الإيجابي.

يعالج المصفوفة من اليسار إلى اليمين، مولّدًا صورة منفصلة لكل عنصر في المصفوفة. في هذه الحالة سيتم توليد 3 صور، واحدة لكل لون.
قم بزيادة عدد الصور إلى 3 لتوليد جميع المتغيرات الثلاثة.

لا يمكن تداخل المصفوفات، ولكن يمكن استخدام مصفوفات متعددة في نفس التوجيه.
يدعم LoRAs المضمنة كعناصر مصفوفة!

### LoRAs المضمنة

مثال توجيه: `flower <lora:sunflowers:1.2>`

تتم معالجته فقط للتوجيه الإيجابي.

يطبق LoRA على التوجيه. يجب أن يكون ملف LoRA موجودًا في دليل `models/loras`.

## الميزات المتقدمة

[انقر هنا لتصفح الميزات المتقدمة.](https://github.com/lllyasviel/Fooocus/discussions/117)

## التفرعات (Forks)

فيما يلي بعض التفرعات لفوكس:

| تفرعات فوكس |
| - |
| [fenneishi/Fooocus-Control](https://github.com/fenneishi/Fooocus-Control) </br>[runew0lf/RuinedFooocus](https://github.com/runew0lf/RuinedFooocus) </br> [MoonRide303/Fooocus-MRE](https://github.com/MoonRide303/Fooocus-MRE) </br> [mashb1t/Fooocus](https://github.com/mashb1t/Fooocus) </br> وغيرها ... |

## الشكر

جزيل الشكر لـ [twri](https://github.com/twri) و [3Diva](https://github.com/3Diva) و [Marc K3nt3L](https://github.com/K3nt3L) لإنشاء أنماط SDXL إضافية متاحة في فوكس.

يبدأ المشروع من مزيج من قواعد أكواد [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) و [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

أيضًا، شكرًا لـ [daswer123](https://github.com/daswer123) للمساهمة بتكبير/تصغير اللوحة (Canvas Zoom)!

## سجل التحديثات

السجل [هنا](update_log.md).

## التوطين/الترجمة/I18N

يمكنك وضع ملفات json في مجلد `language` لترجمة واجهة المستخدم.

على سبيل المثال، فيما يلي محتوى `Fooocus/language/example.json`:

```json
{
  "Generate": "生成",
  "Input Image": "入力画像",
  "Advanced": "고급",
  "SAI 3D Model": "SAI 3D Modèle"
}
```

إذا أضفت الوسيط `--language example`، سيقرأ فوكس `Fooocus/language/example.json` لترجمة واجهة المستخدم.

على سبيل المثال، يمكنك تحرير السطر الأخير من `run.bat` لويندوز كالتالي

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example

أو `run_anime.bat` كالتالي

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset anime

أو `run_realistic.bat` كالتالي

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset realistic

للترجمة العملية، يمكنك إنشاء ملفك الخاص مثل `Fooocus/language/jp.json` أو `Fooocus/language/cn.json` ثم استخدام العلامة `--language jp` أو `--language cn`. من الواضح أن هذه الملفات غير موجودة الآن. **نحتاج مساعدتك لإنشاء هذه الملفات!**

لاحظ أنه إذا لم يتم تقديم `--language` وفي نفس الوقت كان الملف `Fooocus/language/default.json` موجودًا، سيقوم فوكس دائمًا بتحميل `Fooocus/language/default.json` للترجمة. افتراضيًا، الملف `Fooocus/language/default.json` غير موجود.
