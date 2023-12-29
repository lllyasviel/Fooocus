import gradio as gr
import os
import random

def ui_wildcards_enhance(prompt):
    # 创建一个选项卡 wildcards_enhance 通配符增强
    with gr.Tab(label="wildcards enhance"): # 选项卡标题为 wildcards_enhance
                 
        # 指定文件夹为当前目录下的一个子文件夹wildcards
        wildcards_path = './wildcards'

        # 读取wildcards文件夹内的所有txt文件名，不包含后缀txt
        wildcard_file_names = [os.path.splitext(file_name)[0] for file_name in os.listdir(wildcards_path) if file_name.endswith(".txt")]
        # 给wildcard_file_names列表中的每一个字符串的前面和后面都加上"__"，形成 __xxx__ 这样的字符串
        wildcard_file_names = ["__" + file_name + "__" for file_name in wildcard_file_names]

        with gr.Row():
            # 创建一个点击按钮，标签为"将捏人数据添加到提示框里"
            add_xhox_artist_to_prompt_button = gr.Button(label="将自定义人物添加到提示框里", value="Add to Prompt", scale=2)
            #创建一个复选框，标签为“添加前先清空原有提示词”
            clear_before_add_prompt_checkbox = gr.Checkbox(label="Clear before add", value=True)

        # 定义文件名列表
        wildcard_artist_xhox_file_names = [
            'xhox_shot', 'xhox_gender', 'xhox_region', 'xhox_age', 'xhox_bodytype', 'xhox_angle', 'xhox_job',
            'xhox_faceshape', 'xhox_hairstyle', 'xhox_haircolor', 'xhox_eye', 'xhox_bangs', 'xhox_beard', 'xhox_otherfeatures', 'xhox_faceexp', 
            'xhox_chest', 'xhox_waist','xhox_legs', 'xhox_gesture', 'xhox_pose', 
            'xhox_hanfu', 'xhox_suit', 'xhox_topwear', 'xhox_bottomwear', 'xhox_socks', 'xhox_shoes', 'xhox_accessories', 
            'xhox_lighting', 'xhox_color', 'xhox_camera', 'xhox_quality', 'xhox_artist', 'xhox_preset'
            ]

        # 初始化一个空字典来保存所有的下拉菜单选项
        wildcard_artist_xhox_dropdown_choices = {}

        #定义一个读取函数read_wildcard_artist_xhox_file 读取指定通配符文件，便于之后重复调用
        def read_wildcard_artist_xhox_file(wildcard_artist_xhox_file_names):
            # 循环读取每一个文件
            for x_file_name in wildcard_artist_xhox_file_names:
                with open(os.path.join(wildcards_path, f'{x_file_name}.txt'), 'r') as f:
                    # 保存每一个文件的内容到字典中，文件名作为键，文件内容作为值
                    wildcard_artist_xhox_dropdown_choices[x_file_name] = [line.strip() for line in f.readlines()]

        #执行读取            
        read_wildcard_artist_xhox_file(wildcard_artist_xhox_file_names)
        #打印字典的内容 
        #print(wildcard_artist_xhox_dropdown_choices)

        # 把保存的每个文件内容赋值给下拉菜单的选项
        with gr.Tab(label="General"): # General通用选项卡
            with gr.Row():
                wildcard_xhox_11_dropdown = gr.Dropdown(label="Shot", choices=wildcard_artist_xhox_dropdown_choices['xhox_shot'], value="Portrait", scale=2)
                wildcard_xhox_11_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=1.3, scale=1)

                wildcard_xhox_12_dropdown = gr.Dropdown(label="Gender", choices=wildcard_artist_xhox_dropdown_choices['xhox_gender'], scale=2)
                wildcard_xhox_12_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=1.1, scale=1)

                wildcard_xhox_13_dropdown = gr.Dropdown(label="Region", choices=wildcard_artist_xhox_dropdown_choices['xhox_region'], scale=2)
                wildcard_xhox_13_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=1.1, scale=1)

                wildcard_xhox_14_dropdown = gr.Dropdown(label="Age", choices=wildcard_artist_xhox_dropdown_choices['xhox_age'], scale=2)
                wildcard_xhox_14_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=1.1, scale=1)
                
                wildcard_xhox_15_dropdown = gr.Dropdown(label="Body Type", choices=wildcard_artist_xhox_dropdown_choices['xhox_bodytype'], scale=2)
                wildcard_xhox_15_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=1.1, scale=1)
                
                wildcard_xhox_16_dropdown = gr.Dropdown(label="Angle and Gaze", choices=wildcard_artist_xhox_dropdown_choices['xhox_angle'], scale=2)
                wildcard_xhox_16_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=1.1, scale=1)

                wildcard_xhox_17_dropdown = gr.Dropdown(label="Job", choices=wildcard_artist_xhox_dropdown_choices['xhox_job'], scale=2)
                wildcard_xhox_17_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=1.1, scale=1)
                
        with gr.Tab(label="Face"): # Face捏脸选项卡
            with gr.Row():
                wildcard_xhox_21_dropdown = gr.Dropdown(label="Face Shape", choices=wildcard_artist_xhox_dropdown_choices['xhox_faceshape'], scale=2)
                wildcard_xhox_21_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_22_dropdown = gr.Dropdown(label="Hair Style", choices=wildcard_artist_xhox_dropdown_choices['xhox_hairstyle'], scale=2)
                wildcard_xhox_22_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_23_dropdown = gr.Dropdown(label="Hair Color", choices=wildcard_artist_xhox_dropdown_choices['xhox_haircolor'], scale=2)
                wildcard_xhox_23_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_24_dropdown = gr.Dropdown(label="Eye", choices=wildcard_artist_xhox_dropdown_choices['xhox_eye'], scale=2)
                wildcard_xhox_24_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_25_dropdown = gr.Dropdown(label="Bangs", choices=wildcard_artist_xhox_dropdown_choices['xhox_bangs'], scale=2)
                wildcard_xhox_25_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_26_dropdown = gr.Dropdown(label="Beard", choices=wildcard_artist_xhox_dropdown_choices['xhox_beard'], scale=2)
                wildcard_xhox_26_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_27_dropdown = gr.Dropdown(label="Other Features", choices=wildcard_artist_xhox_dropdown_choices['xhox_otherfeatures'], scale=2)
                wildcard_xhox_27_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_28_dropdown = gr.Dropdown(label="Facial Expression", choices=wildcard_artist_xhox_dropdown_choices['xhox_faceexp'], scale=2)
                wildcard_xhox_28_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)


        with gr.Tab(label="Body"): # Body全身选项卡
            with gr.Row():
                wildcard_xhox_31_dropdown = gr.Dropdown(label="Chest", choices=wildcard_artist_xhox_dropdown_choices['xhox_chest'], scale=2)
                wildcard_xhox_31_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_32_dropdown = gr.Dropdown(label="Waist", choices=wildcard_artist_xhox_dropdown_choices['xhox_waist'], scale=2)
                wildcard_xhox_32_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_33_dropdown = gr.Dropdown(label="Legs", choices=wildcard_artist_xhox_dropdown_choices['xhox_legs'], scale=2)
                wildcard_xhox_33_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_34_dropdown = gr.Dropdown(label="Gesture", choices=wildcard_artist_xhox_dropdown_choices['xhox_gesture'], scale=2)
                wildcard_xhox_34_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_35_dropdown = gr.Dropdown(label="Pose", choices=wildcard_artist_xhox_dropdown_choices['xhox_pose'], scale=2)
                wildcard_xhox_35_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)

        with gr.Tab(label="Attire"): # Attire服装选项卡
            with gr.Row():
                wildcard_xhox_40_dropdown = gr.Dropdown(label="Hanfu", choices=wildcard_artist_xhox_dropdown_choices['xhox_hanfu'], scale=2)
                wildcard_xhox_40_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)

                wildcard_xhox_41_dropdown = gr.Dropdown(label="Suit", choices=wildcard_artist_xhox_dropdown_choices['xhox_suit'], scale=2)
                wildcard_xhox_41_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_42_dropdown = gr.Dropdown(label="Topwear", choices=wildcard_artist_xhox_dropdown_choices['xhox_topwear'], value="vest", scale=2)
                wildcard_xhox_42_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_43_dropdown = gr.Dropdown(label="Bottomwear", choices=wildcard_artist_xhox_dropdown_choices['xhox_bottomwear'], scale=2)
                wildcard_xhox_43_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_44_dropdown = gr.Dropdown(label="Socks", choices=wildcard_artist_xhox_dropdown_choices['xhox_socks'], scale=2)
                wildcard_xhox_44_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_45_dropdown = gr.Dropdown(label="Shoes", choices=wildcard_artist_xhox_dropdown_choices['xhox_shoes'], scale=2)
                wildcard_xhox_45_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_46_dropdown = gr.Dropdown(label="Accessories", choices=wildcard_artist_xhox_dropdown_choices['xhox_accessories'], scale=2)
                wildcard_xhox_46_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)


        with gr.Tab(label="Other"): # Other其他选项卡
            with gr.Row():
                wildcard_xhox_51_dropdown = gr.Dropdown(label="Lighting", choices=wildcard_artist_xhox_dropdown_choices['xhox_lighting'], scale=2)
                wildcard_xhox_51_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_52_dropdown = gr.Dropdown(label="Color", choices=wildcard_artist_xhox_dropdown_choices['xhox_color'], scale=2)
                wildcard_xhox_52_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_53_dropdown = gr.Dropdown(label="Camera Parameters", choices=wildcard_artist_xhox_dropdown_choices['xhox_camera'], scale=2)
                wildcard_xhox_53_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_54_dropdown = gr.Dropdown(label="Quality Words", choices=wildcard_artist_xhox_dropdown_choices['xhox_quality'], scale=2)
                wildcard_xhox_54_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_55_dropdown = gr.Dropdown(label="Artist", choices=wildcard_artist_xhox_dropdown_choices['xhox_artist'], scale=2)
                wildcard_xhox_55_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)
                
                wildcard_xhox_56_dropdown = gr.Dropdown(label="Preset", choices=wildcard_artist_xhox_dropdown_choices['xhox_preset'], scale=2)
                wildcard_xhox_56_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.1, value=0.9, scale=1)

        with gr.Tab(label="Readme"): # Readme说明选项卡
            #创建一个文本说明
            gr.HTML('<div align=center><h3>Wildcards Artist - 通配符艺术捏人工具 V0.91</h3></div><br>这是一个提示词文本连接工具，通过选择各种元素来生成描述人物的提示词。然而，有些元素可能会互相冲突或互相影响，因此在开始之前，建议先取消所有风格样式的选择。此外，为了生成符合预期的画面，还需要考虑风格样式和图像尺寸比例的匹配。<p> 例如：<br>1、马尾通常与侧身视角绑定。 <br> 2、更多的头部细节可能会与肖像镜头绑定，从而与全身视角产生冲突。<br> 3、职业角色的选择可能会极大地影响整体的穿着打扮。<br>4、选择的元素越多，效果可能会越弱，冲突和影响也可能越大。<br>......<br>*灵感来源于“ComfyUI Portrait Master 肖像大师”。<br>*通配符数据收集自网络，包括但不限于青龙圣者、-ZHO-、all in one、oldsix、路过银河、Danbooru等。<br><div align=right>by xhox @NGA</div>')
                     

        #创建一个按钮点击事件处理器，拼接捏人数据提示词并添加到提示框中
        def add_xhox_artist_to_prompt(prompt, clear_before_add_prompt_checkbox, *args):

            # 如果“添加前先清空原有提示词”复选框为选中，则将prompt清空
            if clear_before_add_prompt_checkbox:
                prompt = "" 
                
            # 单独处理"镜头类型 "wildcard_xhox_11_dropdown和wildcard_xhox_11_weight
            x_dropdown = args[0]
            x_weight = args[1]
            if f"{x_dropdown}" != "None" and x_dropdown != "" and f"{x_weight}" != "None" and x_weight != "":
                prompt = f"a masterpiece of ({x_dropdown}:{x_weight}) " + prompt
            
            # 拼接捏人数据提示词并添加到提示框中
            for i in range(2, len(args), 2):
                x_dropdown = args[i]
                x_weight = args[i+1]
                if f"{x_dropdown}" != "None" and x_dropdown != "" and f"{x_weight}" != "None" and x_weight != "":
                    prompt += f", ({x_dropdown}:{x_weight})"
            return prompt

        # 设置按钮点击事件处理器，接受参数，并执行函数，最后返回 prompt，把捏人数据提示词添加到提示框中
        add_xhox_artist_to_prompt_button.click(add_xhox_artist_to_prompt, inputs=[
            prompt, clear_before_add_prompt_checkbox,
            wildcard_xhox_11_dropdown, wildcard_xhox_11_weight,
            wildcard_xhox_12_dropdown, wildcard_xhox_12_weight,
            wildcard_xhox_13_dropdown, wildcard_xhox_13_weight,
            wildcard_xhox_14_dropdown, wildcard_xhox_14_weight,
            wildcard_xhox_15_dropdown, wildcard_xhox_15_weight,
            wildcard_xhox_16_dropdown, wildcard_xhox_16_weight,
            wildcard_xhox_17_dropdown, wildcard_xhox_17_weight,

            wildcard_xhox_21_dropdown, wildcard_xhox_21_weight,
            wildcard_xhox_22_dropdown, wildcard_xhox_22_weight,
            wildcard_xhox_23_dropdown, wildcard_xhox_23_weight,
            wildcard_xhox_24_dropdown, wildcard_xhox_24_weight,
            wildcard_xhox_25_dropdown, wildcard_xhox_25_weight,
            wildcard_xhox_26_dropdown, wildcard_xhox_26_weight,
            wildcard_xhox_27_dropdown, wildcard_xhox_27_weight,
            wildcard_xhox_28_dropdown, wildcard_xhox_28_weight,

            wildcard_xhox_31_dropdown, wildcard_xhox_31_weight,
            wildcard_xhox_32_dropdown, wildcard_xhox_32_weight,
            wildcard_xhox_33_dropdown, wildcard_xhox_33_weight,
            wildcard_xhox_34_dropdown, wildcard_xhox_34_weight,
            wildcard_xhox_35_dropdown, wildcard_xhox_35_weight,

            wildcard_xhox_40_dropdown, wildcard_xhox_40_weight,
            wildcard_xhox_41_dropdown, wildcard_xhox_41_weight,
            wildcard_xhox_42_dropdown, wildcard_xhox_42_weight,
            wildcard_xhox_43_dropdown, wildcard_xhox_43_weight,
            wildcard_xhox_44_dropdown, wildcard_xhox_44_weight,
            wildcard_xhox_45_dropdown, wildcard_xhox_45_weight,
            wildcard_xhox_46_dropdown, wildcard_xhox_46_weight,

            wildcard_xhox_51_dropdown, wildcard_xhox_51_weight,
            wildcard_xhox_52_dropdown, wildcard_xhox_52_weight,
            wildcard_xhox_53_dropdown, wildcard_xhox_53_weight,
            wildcard_xhox_54_dropdown, wildcard_xhox_54_weight,
            wildcard_xhox_55_dropdown, wildcard_xhox_55_weight,
            wildcard_xhox_56_dropdown, wildcard_xhox_56_weight

            # 你可以继续添加更多的下拉菜单和权重
        ], outputs=[prompt])


        with gr.Row():
            # 创建一个点击按钮，标签为"给我灵感，随机捏人"，先清空原有提示词，再将随机预设添加到提示框中
            add_randompreset_xhox_to_prompt_button = gr.Button(label="给我灵感，随机捏人", value="Randomize Character", scale=3)

            # 创建一个点击按钮，标签为"重新读取通配符文件内容，刷新选项"（未完成）
            #wildcard_xhox_refresh_button = gr.Button(label="刷新捏人选项", value="刷新选项", scale=1, visible=False)


        # 创建一个按钮点击事件处理器，点击“随机捏人”按钮，将__xhox_preset__预设中的随机一行提示词添加到提示框中
        def add_random_xhox_to_prompt(prompt):
            # 先清空原有提示词，再将预设文件中的随机一行提示词添加到提示框中
            prompt = "---" + random.choice(wildcard_artist_xhox_dropdown_choices['xhox_preset'])
            return prompt

        # 设置按钮点击事件处理器，点击，接受参数，执行函数，返回prompt，将 __通配符__ 添加到提示框
        add_randompreset_xhox_to_prompt_button.click(add_random_xhox_to_prompt, inputs=[prompt],outputs=[prompt])


        
        with gr.Row():
            # 创建一个下拉菜单”选择通配符文件“，选项为读取的通配符文件夹里的txt文件名，不包含后缀.txt
            wildcard_file_names_dropdown = gr.Dropdown(label="Select Wildcard File", choices=wildcard_file_names, value="__color__", scale=2)
            
            # 创建一个点击按钮，标签为"添加通配符到提示框里"
            add_wildcard_file_name_to_prompt_button = gr.Button(label="添加通配符到提示框里", value="Add Wildcard to Prompt", scale=1) 

        # 创建一个按钮点击事件处理器，获取下拉菜单选中的值，然后将结果赋值给提示框里的文本
        def add_wildcard_file_name_to_prompt(prompt,wildcard_file_names_dropdown):
            # 把获取的下拉菜单选项值结果添加到提示框中
            prompt += f", {wildcard_file_names_dropdown}"
            return prompt

        # 设置按钮点击事件处理器，点击，接受参数，执行函数，返回prompt，将 __通配符__ 添加到提示框
        add_wildcard_file_name_to_prompt_button.click(add_wildcard_file_name_to_prompt, inputs=[prompt,wildcard_file_names_dropdown],outputs=[prompt])
