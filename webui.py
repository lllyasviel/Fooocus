import gradio as gr
import random
import os
import json
import time
import shared
import modules.config
import fooocus_version
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import modules.gradio_hijack as grh
import modules.style_sorter as style_sorter
import modules.meta_parser
import args_manager
import copy
import launch
from extras.inpaint_mask import SAMOptions

from modules.sdxl_styles import legal_style_names
from modules.private_logger import get_current_html_path
from modules.ui_gradio_extensions import reload_javascript
from modules.auth import auth_enabled, check_auth
from modules.util import is_json

# ──────────────────────────────────────────────────────────────────
# UI Settings (保存先など永続設定)
# ──────────────────────────────────────────────────────────────────
_UI_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui_settings.json')


def _ui_cfg_load() -> dict:
    if os.path.exists(_UI_SETTINGS_PATH):
        try:
            with open(_UI_SETTINGS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _ui_cfg_save(d: dict):
    with open(_UI_SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


# 起動時に保存先を復元
_ui_cfg = _ui_cfg_load()
if 'path_outputs' in _ui_cfg:
    import modules.config as _cfg_mod
    _cfg_mod.path_outputs = _ui_cfg['path_outputs']
    os.makedirs(_ui_cfg['path_outputs'], exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# User Setting Presets helpers
# ──────────────────────────────────────────────────────────────────
_USER_SP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_setting_presets.json')


def _sp_load() -> dict:
    if os.path.exists(_USER_SP_PATH):
        try:
            with open(_USER_SP_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _sp_save(presets: dict):
    with open(_USER_SP_PATH, 'w', encoding='utf-8') as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)


def _sp_aspect_to_resolution(aspect_label: str) -> str:
    """'1024×1024 | 1:1' → '(1024, 1024)'"""
    try:
        part = aspect_label.split('|')[0].strip()
        w, h = part.split('×')
        return f'({int(w.strip())}, {int(h.strip())})'
    except Exception:
        return '(1024, 1024)'


def get_task(*args):
    args = list(args)
    args.pop(0)

    return worker.AsyncTask(args=args)

def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False
    # outputs=[progress_html, progress_window, progress_gallery, gallery, gallery_paths]

    if len(task.args) == 0:
        return

    execution_start_time = time.perf_counter()
    finished = False

    yield gr.update(visible=True, value=modules.html.make_progress_html(1, 'Waiting for task to start ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False, value=None), \
        gr.update(visible=False), \
        gr.update()

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':

                # help bad internet connection by skipping duplicated preview
                if len(task.yields) > 0:  # if we have the next item
                    if task.yields[0][0] == 'preview':   # if the next item is also a preview
                        # print('Skipped one preview for better internet connection.')
                        continue

                percentage, title, image = product
                yield gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(), \
                    gr.update(visible=False), \
                    gr.update()
            if flag == 'results':
                yield gr.update(visible=True), \
                    gr.update(visible=True), \
                    gr.update(visible=True, value=product), \
                    gr.update(visible=False), \
                    gr.update()
            if flag == 'finish':
                if not args_manager.args.disable_enhance_output_sorting:
                    product = sort_enhance_images(product, task)

                yield gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product), \
                    product
                finished = True

                # delete Fooocus temp images, only keep gradio temp images
                if args_manager.args.disable_image_log:
                    for filepath in product:
                        if isinstance(filepath, str) and os.path.exists(filepath):
                            os.remove(filepath)

    execution_time = time.perf_counter() - execution_start_time
    print(f'Total time: {execution_time:.2f} seconds')
    return


def sort_enhance_images(images, task):
    if not task.should_enhance or len(images) <= task.images_to_enhance_count:
        return images

    sorted_images = []
    walk_index = task.images_to_enhance_count

    for index, enhanced_img in enumerate(images[:task.images_to_enhance_count]):
        sorted_images.append(enhanced_img)
        if index not in task.enhance_stats:
            continue
        target_index = walk_index + task.enhance_stats[index]
        if walk_index < len(images) and target_index <= len(images):
            sorted_images += images[walk_index:target_index]
        walk_index += task.enhance_stats[index]

    return sorted_images


def inpaint_mode_change(mode, inpaint_engine_version):
    assert mode in modules.flags.inpaint_options

    # inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
    # inpaint_disable_initial_latent, inpaint_engine,
    # inpaint_strength, inpaint_respective_field

    if mode == modules.flags.inpaint_option_detail:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=True, samples=modules.config.example_inpaint_prompts),
            False, 'None', 0.5, 0.0
        ]

    if inpaint_engine_version == 'empty':
        inpaint_engine_version = modules.config.default_inpaint_engine_version

    if mode == modules.flags.inpaint_option_modify:
        return [
            gr.update(visible=True), gr.update(visible=False, value=[]),
            gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
            True, inpaint_engine_version, 1.0, 0.0
        ]

    return [
        gr.update(visible=False, value=''), gr.update(visible=True),
        gr.Dataset.update(visible=False, samples=modules.config.example_inpaint_prompts),
        False, inpaint_engine_version, 1.0, 0.618
    ]


reload_javascript()

title = f'Fooocus {fooocus_version.version}'

if isinstance(args_manager.args.preset, str):
    title += ' ' + args_manager.args.preset

shared.gradio_root = gr.Blocks(title=title).queue()

with shared.gradio_root:
    currentTask = gr.State(worker.AsyncTask(args=[]))
    inpaint_engine_state = gr.State('empty')
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                progress_window = grh.Image(label='プレビュー', show_label=True, visible=False, height=768,
                                            elem_classes=['main_view'])
                progress_gallery = gr.Gallery(label='生成完了', show_label=True, object_fit='contain',
                                              height=768, visible=False, elem_classes=['main_view', 'image_gallery'])
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False,
                                    elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='ギャラリー', show_label=False, object_fit='contain', visible=True, height=768,
                                 elem_classes=['resizable_area', 'main_view', 'final_gallery', 'image_gallery'],
                                 elem_id='final_gallery')

            # ── ブラッシュアップ & OpenCV エディットパネル ──
            brushup_selected = gr.State(None)
            brushup_original = gr.State(None)
            gallery_paths = gr.State([])
            with gr.Row(elem_id='brushup_bar'):
                with gr.Column(scale=0, min_width=114, elem_classes='brushup_btn_col', visible=False) as brushup_btn_col:
                    brushup_subtle_btn  = gr.Button('🎨 微調整',      variant='secondary', min_width=0)
                    brushup_strong_btn  = gr.Button('🎭 強変更',      variant='secondary', min_width=0)
                    brushup_upscale_btn = gr.Button('🔍 ×2拡大',     variant='secondary', min_width=0)
                    brushup_gen_btn     = gr.Button('⚡ 変更して生成', variant='primary',   min_width=0)
                with gr.Column(scale=1):
                    brushup_status_html = gr.HTML(
                        value='<span style="color:#5a5a7a;font-size:11px;font-family:monospace;">'
                              '💡 ギャラリーの画像をクリックして選択 → ブラッシュアップ / 編集</span>',
                        elem_id='brushup_status_html')
                    with gr.Row(visible=False, elem_id='edit_panel') as edit_panel:
                        with gr.Column(scale=1):
                            # ── プリセット ──
                            with gr.Row(elem_id='edit_presets'):
                                edit_preset_vivid  = gr.Button('🔆 ビビッド', min_width=0, elem_classes='preset_btn')
                                edit_preset_cinema = gr.Button('🎬 シネマ',   min_width=0, elem_classes='preset_btn')
                                edit_preset_warm   = gr.Button('🌅 ウォーム', min_width=0, elem_classes='preset_btn')
                                edit_preset_cool   = gr.Button('❄️ クール',   min_width=0, elem_classes='preset_btn')
                                edit_preset_soft   = gr.Button('🌫️ ソフト',   min_width=0, elem_classes='preset_btn')
                            # ── スライダー ──
                            edit_brightness  = gr.Slider(label='明るさ',           minimum=-100, maximum=100,  value=0,   step=1)
                            edit_contrast    = gr.Slider(label='コントラスト',      minimum=0.1,  maximum=3.0,  value=1.0, step=0.05)
                            edit_saturation  = gr.Slider(label='彩度',             minimum=0.0,  maximum=3.0,  value=1.0, step=0.05)
                            edit_hue         = gr.Slider(label='色相シフト',        minimum=-90,  maximum=90,   value=0,   step=1)
                            edit_temperature = gr.Slider(label='色温度 ❄→🌅',      minimum=-100, maximum=100,  value=0,   step=1)
                            edit_sharpness   = gr.Slider(label='シャープネス',      minimum=0.0,  maximum=3.0,  value=1.0, step=0.1)
                            # ── 回転・反転 ──
                            with gr.Row():
                                edit_rot_l  = gr.Button('↺ 左90°',  min_width=0)
                                edit_rot_r  = gr.Button('↻ 右90°',  min_width=0)
                                edit_flip_h = gr.Button('↔ 水平反転', min_width=0)
                                edit_flip_v = gr.Button('↕ 垂直反転', min_width=0)
                            # ── アクション ──
                            with gr.Row():
                                edit_reset_btn = gr.Button('🔄 リセット',         min_width=0)
                                edit_apply_btn = gr.Button('✓ 適用 (選択更新)', variant='primary', min_width=0)
                        with gr.Column(scale=1):
                            edit_preview = gr.Image(label='編集プレビュー', type='numpy', interactive=False, height=340)
                            edit_ba_btn  = gr.Button('🔍 元画像を確認 (B/A)', min_width=0)

            with gr.Row():
                with gr.Column(scale=17):
                    prompt = gr.Textbox(show_label=False, placeholder="プロンプトを入力、またはパラメータを貼り付け", elem_id='positive_prompt',
                                        autofocus=True, lines=3)

                    default_prompt = modules.config.default_prompt
                    if isinstance(default_prompt, str) and default_prompt != '':
                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)

                with gr.Column(scale=3, min_width=0):
                    generate_button = gr.Button(label="生成", value="生成", elem_classes='type_row', elem_id='generate_button', visible=True)
                    reset_button = gr.Button(label="再接続", value="再接続", elem_classes='type_row', elem_id='reset_button', visible=False)
                    load_parameter_button = gr.Button(label="パラメータ読込", value="パラメータ読込", elem_classes='type_row', elem_id='load_parameter_button', visible=False)
                    skip_button = gr.Button(label="スキップ", value="スキップ", elem_classes='type_row_half', elem_id='skip_button', visible=False)
                    stop_button = gr.Button(label="停止", value="停止", elem_classes='type_row_half', elem_id='stop_button', visible=False)

                    def stop_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        currentTask.last_stop = 'stop'
                        if (currentTask.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    def skip_clicked(currentTask):
                        import ldm_patched.modules.model_management as model_management
                        currentTask.last_stop = 'skip'
                        if (currentTask.processing):
                            model_management.interrupt_current_processing()
                        return currentTask

                    stop_button.click(stop_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False, _js='cancelGenerateForever')
                    skip_button.click(skip_clicked, inputs=currentTask, outputs=currentTask, queue=False, show_progress=False)
            with gr.Accordion('🔮 Prompt Forge — タグ選択＆翻訳', open=False, elem_id='pf_accordion'):
                gr.HTML(value=(
                    '<div id="pf_container" style="position:relative;height:480px;'
                    'border-radius:6px;overflow:hidden;background:#0a0a0f;">'
                    '  <div id="pf_loading" style="position:absolute;inset:0;display:flex;'
                    '    align-items:center;justify-content:center;flex-direction:column;gap:12px;'
                    '    background:#0a0a0f;color:#7c4dff;font-family:monospace;font-size:13px;'
                    '    pointer-events:none;z-index:2;">'
                    '    <div style="width:36px;height:36px;border:3px solid #2a2a45;'
                    '      border-top-color:#7c4dff;border-radius:50%;'
                    '      animation:pf-spin 0.8s linear infinite;"></div>'
                    '    <span>Prompt Forge を読み込み中...</span>'
                    '  </div>'
                    '  <style>@keyframes pf-spin{to{transform:rotate(360deg)}}</style>'
                    '  <iframe id="pf_iframe" src="http://127.0.0.1:8080/?embed=1"'
                    '    style="width:100%;height:100%;border:none;background:#0a0a0f;"'
                    '    allow="clipboard-write"'
                    '    onload="var l=document.getElementById(\'pf_loading\');if(l)l.style.display=\'none\';">'
                    '  </iframe>'
                    '</div>'
                    '<div style="text-align:right;margin-top:4px;">'
                    '  <a href="#" onclick="window.scrollTo({top:0,behavior:\'smooth\'});return false;"'
                    '     style="font-size:12px;color:#7c4dff;text-decoration:none;opacity:0.8;">'
                    '    ▲ 生成エリアへ戻る'
                    '  </a>'
                    '</div>'
                ))
            with gr.Row(elem_classes='advanced_check_row'):
                input_image_checkbox = gr.Checkbox(label='入力画像', value=modules.config.default_image_prompt_checkbox, container=False, elem_classes='min_check')
                enhance_checkbox = gr.Checkbox(label='強化', value=modules.config.default_enhance_checkbox, container=False, elem_classes='min_check')
                advanced_checkbox = gr.Checkbox(label='詳細設定', value=modules.config.default_advanced_checkbox, container=False, elem_classes='min_check')
            with gr.Row(visible=modules.config.default_image_prompt_checkbox) as image_input_panel:
                with gr.Tabs(selected=modules.config.default_selected_image_input_tab_id) as image_input_tabs:
                    with gr.Tab(label='拡大・バリエーション', id='uov_tab') as uov_tab:
                        with gr.Row():
                            with gr.Column():
                                uov_input_image = grh.Image(label='画像', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                uov_method = gr.Radio(label='拡大またはバリエーション:', choices=flags.uov_list, value=modules.config.default_uov_method)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/390" target="_blank">📖 ドキュメント</a>')
                    with gr.Tab(label='画像プロンプト', id='ip_tab') as ip_tab:
                        with gr.Row():
                            ip_images = []
                            ip_types = []
                            ip_stops = []
                            ip_weights = []
                            ip_ctrls = []
                            ip_ad_cols = []
                            for image_count in range(modules.config.default_controlnet_image_count):
                                image_count += 1
                                with gr.Column():
                                    ip_image = grh.Image(label='画像', source='upload', type='numpy', show_label=False, height=300, value=modules.config.default_ip_images[image_count])
                                    ip_images.append(ip_image)
                                    ip_ctrls.append(ip_image)
                                    with gr.Column(visible=modules.config.default_image_prompt_advanced_checkbox) as ad_col:
                                        with gr.Row():
                                            ip_stop = gr.Slider(label='停止位置', minimum=0.0, maximum=1.0, step=0.001, value=modules.config.default_ip_stop_ats[image_count])
                                            ip_stops.append(ip_stop)
                                            ip_ctrls.append(ip_stop)

                                            ip_weight = gr.Slider(label='ウェイト', minimum=0.0, maximum=2.0, step=0.001, value=modules.config.default_ip_weights[image_count])
                                            ip_weights.append(ip_weight)
                                            ip_ctrls.append(ip_weight)

                                        ip_type = gr.Radio(label='タイプ', choices=flags.ip_list, value=modules.config.default_ip_types[image_count], container=False)
                                        ip_types.append(ip_type)
                                        ip_ctrls.append(ip_type)

                                        ip_type.change(lambda x: flags.default_parameters[x], inputs=[ip_type], outputs=[ip_stop, ip_weight], queue=False, show_progress=False)
                                    ip_ad_cols.append(ad_col)
                        ip_advanced = gr.Checkbox(label='詳細設定', value=modules.config.default_image_prompt_advanced_checkbox, container=False)
                        gr.HTML('* 「画像プロンプト」は Fooocus Image Mixture Engine (v1.0.1) で動作しています。<a href="https://github.com/lllyasviel/Fooocus/discussions/557" target="_blank">📖 ドキュメント</a>')

                        def ip_advance_checked(x):
                            return [gr.update(visible=x)] * len(ip_ad_cols) + \
                                [flags.default_ip] * len(ip_types) + \
                                [flags.default_parameters[flags.default_ip][0]] * len(ip_stops) + \
                                [flags.default_parameters[flags.default_ip][1]] * len(ip_weights)

                        ip_advanced.change(ip_advance_checked, inputs=ip_advanced,
                                           outputs=ip_ad_cols + ip_types + ip_stops + ip_weights,
                                           queue=False, show_progress=False)

                    with gr.Tab(label='インペイント・アウトペイント', id='inpaint_tab') as inpaint_tab:
                        with gr.Row():
                            with gr.Column():
                                inpaint_input_image = grh.Image(label='画像', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", elem_id='inpaint_canvas', show_label=False)
                                inpaint_advanced_masking_checkbox = gr.Checkbox(label='高度マスキング機能を有効化', value=modules.config.default_inpaint_advanced_masking_checkbox)
                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='方式')
                                inpaint_additional_prompt = gr.Textbox(placeholder="インペイントしたい内容を入力", elem_id='inpaint_additional_prompt', label='インペイント追加プロンプト', visible=False)
                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='アウトペイント方向')
                                example_inpaint_prompts = gr.Dataset(samples=modules.config.example_inpaint_prompts,
                                                                     label='追加プロンプト クイックリスト',
                                                                     components=[inpaint_additional_prompt],
                                                                     visible=False)
                                gr.HTML('* Fooocus Inpaint Engine で動作しています。<a href="https://github.com/lllyasviel/Fooocus/discussions/414" target="_blank">📖 ドキュメント</a>')
                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)

                            with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:
                                inpaint_mask_image = grh.Image(label='マスクアップロード', source='upload', type='numpy', tool='sketch', height=500, brush_color="#FFFFFF", mask_opacity=1, elem_id='inpaint_mask_canvas')
                                invert_mask_checkbox = gr.Checkbox(label='生成時にマスクを反転', value=modules.config.default_invert_mask_checkbox)
                                inpaint_mask_model = gr.Dropdown(label='マスク生成モデル',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_inpaint_mask_model)
                                inpaint_mask_cloth_category = gr.Dropdown(label='衣類カテゴリー',
                                                             choices=flags.inpaint_mask_cloth_category,
                                                             value=modules.config.default_inpaint_mask_cloth_category,
                                                             visible=False)
                                inpaint_mask_dino_prompt_text = gr.Textbox(label='検出プロンプト', value='', visible=False, info='できるだけ単数形で入力してください', placeholder='検出したい対象を入力')
                                example_inpaint_mask_dino_prompt_text = gr.Dataset(
                                    samples=modules.config.example_enhance_detection_prompts,
                                    label='検出プロンプト クイックリスト',
                                    components=[inpaint_mask_dino_prompt_text],
                                    visible=modules.config.default_inpaint_mask_model == 'sam')
                                example_inpaint_mask_dino_prompt_text.click(lambda x: x[0],
                                                                            inputs=example_inpaint_mask_dino_prompt_text,
                                                                            outputs=inpaint_mask_dino_prompt_text,
                                                                            show_progress=False, queue=False)

                                with gr.Accordion("詳細オプション", visible=False, open=False) as inpaint_mask_advanced_options:
                                    inpaint_mask_sam_model = gr.Dropdown(label='SAM モデル', choices=flags.inpaint_mask_sam_model, value=modules.config.default_inpaint_mask_sam_model)
                                    inpaint_mask_box_threshold = gr.Slider(label="ボックス閾値", minimum=0.0, maximum=1.0, value=0.3, step=0.05)
                                    inpaint_mask_text_threshold = gr.Slider(label="テキスト閾値", minimum=0.0, maximum=1.0, value=0.25, step=0.05)
                                    inpaint_mask_sam_max_detections = gr.Slider(label="最大検出数", info="0 に設定するとすべて検出", minimum=0, maximum=10, value=modules.config.default_sam_max_detections, step=1, interactive=True)
                                generate_mask_button = gr.Button(value='画像からマスクを生成')

                                def generate_mask(image, mask_model, cloth_category, dino_prompt_text, sam_model, box_threshold, text_threshold, sam_max_detections, dino_erode_or_dilate, dino_debug):
                                    from extras.inpaint_mask import generate_mask_from_image

                                    extras = {}
                                    sam_options = None
                                    if mask_model == 'u2net_cloth_seg':
                                        extras['cloth_category'] = cloth_category
                                    elif mask_model == 'sam':
                                        sam_options = SAMOptions(
                                            dino_prompt=dino_prompt_text,
                                            dino_box_threshold=box_threshold,
                                            dino_text_threshold=text_threshold,
                                            dino_erode_or_dilate=dino_erode_or_dilate,
                                            dino_debug=dino_debug,
                                            max_detections=sam_max_detections,
                                            model_type=sam_model
                                        )

                                    mask, _, _, _ = generate_mask_from_image(image, mask_model, extras, sam_options)

                                    return mask


                                inpaint_mask_model.change(lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                                                    [gr.update(visible=x == 'sam')] * 2 +
                                                                    [gr.Dataset.update(visible=x == 'sam',
                                                                                       samples=modules.config.example_enhance_detection_prompts)],
                                                          inputs=inpaint_mask_model,
                                                          outputs=[inpaint_mask_cloth_category,
                                                                   inpaint_mask_dino_prompt_text,
                                                                   inpaint_mask_advanced_options,
                                                                   example_inpaint_mask_dino_prompt_text],
                                                          queue=False, show_progress=False)

                    with gr.Tab(label='画像解析', id='describe_tab') as describe_tab:
                        with gr.Row():
                            with gr.Column():
                                describe_input_image = grh.Image(label='画像', source='upload', type='numpy', show_label=False)
                            with gr.Column():
                                describe_methods = gr.CheckboxGroup(
                                    label='コンテンツタイプ',
                                    choices=flags.describe_types,
                                    value=modules.config.default_describe_content_type)
                                describe_apply_styles = gr.Checkbox(label='スタイルを適用', value=modules.config.default_describe_apply_prompts_checkbox)
                                describe_btn = gr.Button(value='画像をプロンプトに変換')
                                describe_image_size = gr.Textbox(label='画像サイズと推奨サイズ', elem_id='describe_image_size', visible=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/1363" target="_blank">📖 ドキュメント</a>')

                                def trigger_show_image_properties(image):
                                    value = modules.util.get_image_size_info(image, modules.flags.sdxl_aspect_ratios)
                                    return gr.update(value=value, visible=True)

                                describe_input_image.upload(trigger_show_image_properties, inputs=describe_input_image,
                                                            outputs=describe_image_size, show_progress=False, queue=False)

                    with gr.Tab(label='強化', id='enhance_tab') as enhance_tab:
                        with gr.Row():
                            with gr.Column():
                                enhance_input_image = grh.Image(label='強化モード用（画像生成をスキップ）', source='upload', type='numpy')
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">📖 ドキュメント</a>')

                    with gr.Tab(label='メタデータ', id='metadata_tab') as metadata_tab:
                        with gr.Column():
                            metadata_input_image = grh.Image(label='Fooocus で作成した画像を読み込む', source='upload', type='pil')
                            metadata_json = gr.JSON(label='メタデータ')
                            metadata_import_button = gr.Button(value='メタデータを適用')

                        def trigger_metadata_preview(file):
                            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)

                            results = {}
                            if parameters is not None:
                                results['parameters'] = parameters

                            if isinstance(metadata_scheme, flags.MetadataScheme):
                                results['metadata_scheme'] = metadata_scheme.value

                            return results

                        metadata_input_image.upload(trigger_metadata_preview, inputs=metadata_input_image,
                                                    outputs=metadata_json, queue=False, show_progress=True)

            with gr.Row(visible=modules.config.default_enhance_checkbox) as enhance_input_panel:
                with gr.Tabs():
                    with gr.Tab(label='拡大・バリエーション'):
                        with gr.Row():
                            with gr.Column():
                                enhance_uov_method = gr.Radio(label='拡大またはバリエーション:', choices=flags.uov_list,
                                                              value=modules.config.default_enhance_uov_method)
                                enhance_uov_processing_order = gr.Radio(label='処理順序',
                                                                        info='「前」は細部の強化、「後」は広い領域の強化に使用します。',
                                                                        choices=flags.enhancement_uov_processing_order,
                                                                        value=modules.config.default_enhance_uov_processing_order)
                                enhance_uov_prompt_type = gr.Radio(label='プロンプト',
                                                                   info='拡大またはバリエーションに使用するプロンプトを選択します。',
                                                                   choices=flags.enhancement_uov_prompt_types,
                                                                   value=modules.config.default_enhance_uov_prompt_type,
                                                                   visible=modules.config.default_enhance_uov_processing_order == flags.enhancement_uov_after)

                                enhance_uov_processing_order.change(lambda x: gr.update(visible=x == flags.enhancement_uov_after),
                                                                    inputs=enhance_uov_processing_order,
                                                                    outputs=enhance_uov_prompt_type,
                                                                    queue=False, show_progress=False)
                                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">📖 ドキュメント</a>')
                    enhance_ctrls = []
                    enhance_inpaint_mode_ctrls = []
                    enhance_inpaint_engine_ctrls = []
                    enhance_inpaint_update_ctrls = []
                    for index in range(modules.config.default_enhance_tabs):
                        with gr.Tab(label=f'#{index + 1}') as enhance_tab_item:
                            enhance_enabled = gr.Checkbox(label='有効化', value=False, elem_classes='min_check',
                                                          container=False)

                            enhance_mask_dino_prompt_text = gr.Textbox(label='検出プロンプト',
                                                                       info='できるだけ単数形で入力してください',
                                                                       placeholder='検出したい対象を入力',
                                                                       interactive=True,
                                                                       visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text = gr.Dataset(
                                samples=modules.config.example_enhance_detection_prompts,
                                label='検出プロンプト クイックリスト',
                                components=[enhance_mask_dino_prompt_text],
                                visible=modules.config.default_enhance_inpaint_mask_model == 'sam')
                            example_enhance_mask_dino_prompt_text.click(lambda x: x[0],
                                                                        inputs=example_enhance_mask_dino_prompt_text,
                                                                        outputs=enhance_mask_dino_prompt_text,
                                                                        show_progress=False, queue=False)

                            enhance_prompt = gr.Textbox(label="強化用ポジティブプロンプト",
                                                        placeholder="空欄の場合は元のプロンプトを使用します。",
                                                        elem_id='enhance_prompt')
                            enhance_negative_prompt = gr.Textbox(label="強化用ネガティブプロンプト",
                                                                 placeholder="空欄の場合は元のネガティブプロンプトを使用します。",
                                                                 elem_id='enhance_negative_prompt')

                            with gr.Accordion("検出", open=False):
                                enhance_mask_model = gr.Dropdown(label='マスク生成モデル',
                                                                 choices=flags.inpaint_mask_models,
                                                                 value=modules.config.default_enhance_inpaint_mask_model)
                                enhance_mask_cloth_category = gr.Dropdown(label='衣類カテゴリー',
                                                                          choices=flags.inpaint_mask_cloth_category,
                                                                          value=modules.config.default_inpaint_mask_cloth_category,
                                                                          visible=modules.config.default_enhance_inpaint_mask_model == 'u2net_cloth_seg',
                                                                          interactive=True)

                                with gr.Accordion("SAM オプション",
                                                  visible=modules.config.default_enhance_inpaint_mask_model == 'sam',
                                                  open=False) as sam_options:
                                    enhance_mask_sam_model = gr.Dropdown(label='SAM モデル',
                                                                         choices=flags.inpaint_mask_sam_model,
                                                                         value=modules.config.default_inpaint_mask_sam_model,
                                                                         interactive=True)
                                    enhance_mask_box_threshold = gr.Slider(label="ボックス閾値", minimum=0.0,
                                                                           maximum=1.0, value=0.3, step=0.05,
                                                                           interactive=True)
                                    enhance_mask_text_threshold = gr.Slider(label="テキスト閾値", minimum=0.0,
                                                                            maximum=1.0, value=0.25, step=0.05,
                                                                            interactive=True)
                                    enhance_mask_sam_max_detections = gr.Slider(label="最大検出数",
                                                                                info="0 に設定するとすべて検出",
                                                                                minimum=0, maximum=10,
                                                                                value=modules.config.default_sam_max_detections,
                                                                                step=1, interactive=True)

                            with gr.Accordion("インペイント", visible=True, open=False):
                                enhance_inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options,
                                                                   value=modules.config.default_inpaint_method,
                                                                   label='方式', interactive=True)
                                enhance_inpaint_disable_initial_latent = gr.Checkbox(
                                    label='インペイントの初期潜在を無効化', value=False)
                                enhance_inpaint_engine = gr.Dropdown(label='インペイントエンジン',
                                                                     value=modules.config.default_inpaint_engine_version,
                                                                     choices=flags.inpaint_engine_versions,
                                                                     info='Fooocus インペイントモデルのバージョン。Quality または Speed を推奨。')
                                enhance_inpaint_strength = gr.Slider(label='インペイント デノイズ強度',
                                                                     minimum=0.0, maximum=1.0, step=0.001,
                                                                     value=1.0,
                                                                     info='A1111 インペイントのデノイズ強度と同じです。'
                                                                          'インペイントのみに使用（アウトペイントには無効）。'
                                                                          '（アウトペイントは常に 1.0）')
                                enhance_inpaint_respective_field = gr.Slider(label='インペイント対象範囲',
                                                                             minimum=0.0, maximum=1.0, step=0.001,
                                                                             value=0.618,
                                                                             info='インペイントする領域。'
                                                                                  '0 は A1111 の「マスクのみ」、'
                                                                                  '1 は「画像全体」と同じです。'
                                                                                  'インペイントのみ有効（アウトペイントは常に 1.0）')
                                enhance_inpaint_erode_or_dilate = gr.Slider(label='マスク 侵食/膨張',
                                                                            minimum=-64, maximum=64, step=1, value=0,
                                                                            info='正の値はマスクの白い領域を拡大、'
                                                                                 '負の値は縮小します。'
                                                                                 '（デフォルト 0、マスク反転前に処理）')
                                enhance_mask_invert = gr.Checkbox(label='マスクを反転', value=False)

                            gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/3281" target="_blank">📖 ドキュメント</a>')

                        enhance_ctrls += [
                            enhance_enabled,
                            enhance_mask_dino_prompt_text,
                            enhance_prompt,
                            enhance_negative_prompt,
                            enhance_mask_model,
                            enhance_mask_cloth_category,
                            enhance_mask_sam_model,
                            enhance_mask_text_threshold,
                            enhance_mask_box_threshold,
                            enhance_mask_sam_max_detections,
                            enhance_inpaint_disable_initial_latent,
                            enhance_inpaint_engine,
                            enhance_inpaint_strength,
                            enhance_inpaint_respective_field,
                            enhance_inpaint_erode_or_dilate,
                            enhance_mask_invert
                        ]

                        enhance_inpaint_mode_ctrls += [enhance_inpaint_mode]
                        enhance_inpaint_engine_ctrls += [enhance_inpaint_engine]

                        enhance_inpaint_update_ctrls += [[
                            enhance_inpaint_mode, enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ]]

                        enhance_inpaint_mode.change(inpaint_mode_change, inputs=[enhance_inpaint_mode, inpaint_engine_state], outputs=[
                            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
                            enhance_inpaint_disable_initial_latent, enhance_inpaint_engine,
                            enhance_inpaint_strength, enhance_inpaint_respective_field
                        ], show_progress=False, queue=False)

                        enhance_mask_model.change(
                            lambda x: [gr.update(visible=x == 'u2net_cloth_seg')] +
                                      [gr.update(visible=x == 'sam')] * 2 +
                                      [gr.Dataset.update(visible=x == 'sam',
                                                         samples=modules.config.example_enhance_detection_prompts)],
                            inputs=enhance_mask_model,
                            outputs=[enhance_mask_cloth_category, enhance_mask_dino_prompt_text, sam_options,
                                     example_enhance_mask_dino_prompt_text],
                            queue=False, show_progress=False)

            switch_js = "(x) => {if(x){viewer_to_bottom(100);viewer_to_bottom(500);}else{viewer_to_top();} return x;}"
            down_js = "() => {viewer_to_bottom();}"

            input_image_checkbox.change(lambda x: gr.update(visible=x), inputs=input_image_checkbox,
                                        outputs=image_input_panel, queue=False, show_progress=False, _js=switch_js)
            ip_advanced.change(lambda: None, queue=False, show_progress=False, _js=down_js)

            current_tab = gr.Textbox(value='uov', visible=False)
            uov_tab.select(lambda: 'uov', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            inpaint_tab.select(lambda: 'inpaint', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            ip_tab.select(lambda: 'ip', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            describe_tab.select(lambda: 'desc', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            enhance_tab.select(lambda: 'enhance', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            metadata_tab.select(lambda: 'metadata', outputs=current_tab, queue=False, _js=down_js, show_progress=False)
            enhance_checkbox.change(lambda x: gr.update(visible=x), inputs=enhance_checkbox,
                                        outputs=enhance_input_panel, queue=False, show_progress=False, _js=switch_js)

        with gr.Column(scale=1, visible=modules.config.default_advanced_checkbox) as advanced_column:
            with gr.Tab(label='設定'):
                if not args_manager.args.disable_preset_selection:
                    preset_selection = gr.Dropdown(label='プリセット',
                                                   choices=modules.config.available_presets,
                                                   value=args_manager.args.preset if args_manager.args.preset else "initial",
                                                   interactive=True)

                # ── User Setting Presets UI ──
                with gr.Accordion('⭐ 設定プリセット', open=False, elem_id='user_sp_accordion'):
                    # ── 保存先フォルダ ──
                    with gr.Row():
                        output_path_box = gr.Textbox(
                            label='📁 保存先フォルダ',
                            value=modules.config.path_outputs,
                            scale=5, interactive=True,
                            placeholder='例: C:/Users/xxx/Pictures/Fooocus')
                        output_path_btn = gr.Button('✓ 変更', variant='secondary', scale=1, min_width=0)
                    output_path_status = gr.Markdown(value='')
                    gr.HTML('<hr style="border-color:rgba(255,255,255,0.08);margin:4px 0;">')
                    # ── プリセット操作 ──
                    with gr.Row():
                        user_sp_dropdown = gr.Dropdown(
                            label='保存済みプリセット',
                            choices=list(_sp_load().keys()),
                            value=None, scale=4, interactive=True)
                        user_sp_load_btn = gr.Button('📂 読込', variant='primary', scale=1, min_width=0)
                        user_sp_delete_btn = gr.Button('🗑️ 削除', variant='stop', scale=1, min_width=0)
                    with gr.Row():
                        user_sp_name = gr.Textbox(
                            label='保存名', placeholder='プリセット名を入力して保存...', scale=4)
                        user_sp_save_btn = gr.Button('💾 現在の設定を保存', variant='secondary', scale=2, min_width=0)
                    user_sp_status = gr.Markdown(value='')

                performance_selection = gr.Radio(label='パフォーマンス',
                                                 choices=flags.Performance.values(),
                                                 value=modules.config.default_performance,
                                                 elem_classes=['performance_selection'])

                with gr.Accordion(label='アスペクト比', open=False, elem_id='aspect_ratios_accordion') as aspect_ratios_accordion:
                    aspect_ratios_selection = gr.Radio(label='アスペクト比', show_label=False,
                                                       choices=modules.config.available_aspect_ratios_labels,
                                                       value=modules.config.default_aspect_ratio,
                                                       info='幅 × 高さ',
                                                       elem_classes='aspect_ratios')

                    aspect_ratios_selection.change(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')
                    shared.gradio_root.load(lambda x: None, inputs=aspect_ratios_selection, queue=False, show_progress=False, _js='(x)=>{refresh_aspect_ratios_label(x);}')

                image_number = gr.Slider(label='生成枚数', minimum=1, maximum=modules.config.default_max_image_number, step=1, value=modules.config.default_image_number)

                output_format = gr.Radio(label='出力形式',
                                         choices=flags.OutputFormat.list(),
                                         value=modules.config.default_output_format)

                negative_prompt = gr.Textbox(label='ネガティブプロンプト', show_label=True, placeholder="ここにプロンプトを入力",
                                             info='生成したくない要素を入力してください。', lines=2,
                                             elem_id='negative_prompt',
                                             value=modules.config.default_prompt_negative)
                seed_random = gr.Checkbox(label='ランダム', value=True)
                image_seed = gr.Textbox(label='シード', value=0, max_lines=1, visible=False) # workaround for https://github.com/gradio-app/gradio/issues/5354

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, seed_string):
                    if r:
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)
                    else:
                        try:
                            seed_value = int(seed_string)
                            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                                return seed_value
                        except ValueError:
                            pass
                        return random.randint(constants.MIN_SEED, constants.MAX_SEED)

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed],
                                   queue=False, show_progress=False)

                def update_history_link():
                    if args_manager.args.disable_image_log:
                        return gr.update(value='')

                    return gr.update(value=f'<a href="file={get_current_html_path(output_format)}" target="_blank">📚 生成履歴</a>')

                history_link = gr.HTML()
                shared.gradio_root.load(update_history_link, outputs=history_link, queue=False, show_progress=False)

            with gr.Tab(label='スタイル', elem_classes=['style_selections_tab']):
                style_sorter.try_load_sorted_styles(
                    style_names=legal_style_names,
                    default_selected=modules.config.default_styles)

                style_search_bar = gr.Textbox(show_label=False, container=False,
                                              placeholder="🔍 スタイルを検索...",
                                              value="",
                                              label='スタイル検索')
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=copy.deepcopy(style_sorter.all_styles),
                                                    value=copy.deepcopy(modules.config.default_styles),
                                                    label='選択中のスタイル',
                                                    elem_classes=['style_selections'])
                gradio_receiver_style_selections = gr.Textbox(elem_id='gradio_receiver_style_selections', visible=False)

                shared.gradio_root.load(lambda: gr.update(choices=copy.deepcopy(style_sorter.all_styles)),
                                        outputs=style_selections)

                style_search_bar.change(style_sorter.search_styles,
                                        inputs=[style_selections, style_search_bar],
                                        outputs=style_selections,
                                        queue=False,
                                        show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

                gradio_receiver_style_selections.input(style_sorter.sort_styles,
                                                       inputs=style_selections,
                                                       outputs=style_selections,
                                                       queue=False,
                                                       show_progress=False).then(
                    lambda: None, _js='()=>{refresh_style_localization();}')

            with gr.Tab(label='モデル'):
                with gr.Group():
                    with gr.Row():
                        base_model = gr.Dropdown(label='ベースモデル（SDXL のみ）', choices=modules.config.model_filenames, value=modules.config.default_base_model_name, show_label=True)
                        refiner_model = gr.Dropdown(label='リファイナー（SDXL または SD 1.5）', choices=['None'] + modules.config.model_filenames, value=modules.config.default_refiner_model_name, show_label=True)

                    refiner_switch = gr.Slider(label='リファイナー切替タイミング', minimum=0.1, maximum=1.0, step=0.0001,
                                               info='SD1.5 リアル系: 0.4 / アニメ系: 0.667 / XL リファイナー: 0.8 / 2モデル切替: 任意の値',
                                               value=modules.config.default_refiner_switch,
                                               visible=modules.config.default_refiner_model_name != 'None')

                    refiner_model.change(lambda x: gr.update(visible=x != 'None'),
                                         inputs=refiner_model, outputs=refiner_switch, show_progress=False, queue=False)

                with gr.Group():
                    lora_ctrls = []

                    for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
                        with gr.Row():
                            lora_enabled = gr.Checkbox(label='有効', value=enabled,
                                                       elem_classes=['lora_enable', 'min_check'], scale=1)
                            lora_model = gr.Dropdown(label=f'LoRA {i + 1}',
                                                     choices=['None'] + modules.config.lora_filenames, value=filename,
                                                     elem_classes='lora_model', scale=5)
                            lora_weight = gr.Slider(label='ウェイト', minimum=modules.config.default_loras_min_weight,
                                                    maximum=modules.config.default_loras_max_weight, step=0.01, value=weight,
                                                    elem_classes='lora_weight', scale=5)
                            lora_ctrls += [lora_enabled, lora_model, lora_weight]

                with gr.Row():
                    refresh_files = gr.Button(label='更新', value='🔄 全ファイルを更新', variant='secondary', elem_classes='refresh_button')
            with gr.Tab(label='詳細設定'):
                guidance_scale = gr.Slider(label='ガイダンススケール', minimum=1.0, maximum=30.0, step=0.01,
                                           value=modules.config.default_cfg_scale,
                                           info='値が高いほどスタイルが鮮明・鮮やか・芸術的になります。')
                sharpness = gr.Slider(label='画像シャープネス', minimum=0.0, maximum=30.0, step=0.001,
                                      value=modules.config.default_sample_sharpness,
                                      info='値が高いほど画像とテクスチャがシャープになります。')
                gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117" target="_blank">📖 ドキュメント</a>')
                dev_mode = gr.Checkbox(label='開発者デバッグモード', value=modules.config.default_developer_debug_mode_checkbox, container=False)

                with gr.Column(visible=modules.config.default_developer_debug_mode_checkbox) as dev_tools:
                    with gr.Tab(label='デバッグツール'):
                        adm_scaler_positive = gr.Slider(label='ポジティブ ADM ガイダンス スケーラー', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=1.5, info='ポジティブ ADM に乗算するスケーラー（無効にするには 1.0）。')
                        adm_scaler_negative = gr.Slider(label='ネガティブ ADM ガイダンス スケーラー', minimum=0.1, maximum=3.0,
                                                        step=0.001, value=0.8, info='ネガティブ ADM に乗算するスケーラー（無効にするには 1.0）。')
                        adm_scaler_end = gr.Slider(label='ADM ガイダンス終了ステップ', minimum=0.0, maximum=1.0,
                                                   step=0.001, value=0.3,
                                                   info='ポジティブ/ネガティブ ADM ガイダンスを終了するタイミング。')

                        refiner_swap_method = gr.Dropdown(label='リファイナー スワップ方式', value=flags.refiner_swap_method,
                                                          choices=['joint', 'separate', 'vae'])

                        adaptive_cfg = gr.Slider(label='CFG TSNR ミミッキング', minimum=1.0, maximum=30.0, step=0.01,
                                                 value=modules.config.default_cfg_tsnr,
                                                 info='TSNR 向け CFG ミミッキングを有効化します（実 CFG > ミミッキング CFG のとき有効）。')
                        clip_skip = gr.Slider(label='CLIP スキップ', minimum=1, maximum=flags.clip_skip_max, step=1,
                                                 value=modules.config.default_clip_skip,
                                                 info='過学習を避けるため CLIP レイヤーをスキップします（1: スキップなし、推奨: 2）。')
                        sampler_name = gr.Dropdown(label='サンプラー', choices=flags.sampler_list,
                                                   value=modules.config.default_sampler)
                        scheduler_name = gr.Dropdown(label='スケジューラー', choices=flags.scheduler_list,
                                                     value=modules.config.default_scheduler)
                        vae_name = gr.Dropdown(label='VAE', choices=[modules.flags.default_vae] + modules.config.vae_filenames,
                                                     value=modules.config.default_vae, show_label=True)

                        generate_image_grid = gr.Checkbox(label='バッチごとに画像グリッドを生成',
                                                          info='（実験的）環境によってはパフォーマンスに影響する場合があります。',
                                                          value=False)

                        overwrite_step = gr.Slider(label='サンプリングステップ 強制上書き',
                                                   minimum=-1, maximum=200, step=1,
                                                   value=modules.config.default_overwrite_step,
                                                   info='-1 で無効。開発者向けデバッグ用。')
                        overwrite_switch = gr.Slider(label='リファイナー切替ステップ 強制上書き',
                                                     minimum=-1, maximum=200, step=1,
                                                     value=modules.config.default_overwrite_switch,
                                                     info='-1 で無効。開発者向けデバッグ用。')
                        overwrite_width = gr.Slider(label='生成幅 強制上書き',
                                                    minimum=-1, maximum=2048, step=1, value=-1,
                                                    info='-1 で無効。SDXL の学習外の値は品質が低下します。')
                        overwrite_height = gr.Slider(label='生成高さ 強制上書き',
                                                     minimum=-1, maximum=2048, step=1, value=-1,
                                                     info='-1 で無効。SDXL の学習外の値は品質が低下します。')
                        overwrite_vary_strength = gr.Slider(label='「バリエーション」デノイズ強度 強制上書き',
                                                            minimum=-1, maximum=1.0, step=0.001, value=-1,
                                                            info='負の値で無効。開発者向けデバッグ用。')
                        overwrite_upscale_strength = gr.Slider(label='「アップスケール」デノイズ強度 強制上書き',
                                                               minimum=-1, maximum=1.0, step=0.001,
                                                               value=modules.config.default_overwrite_upscale,
                                                               info='負の値で無効。開発者向けデバッグ用。')

                        disable_preview = gr.Checkbox(label='プレビューを無効化', value=modules.config.default_black_out_nsfw,
                                                      interactive=not modules.config.default_black_out_nsfw,
                                                      info='生成中のプレビュー表示を無効にします。')
                        disable_intermediate_results = gr.Checkbox(label='中間結果を無効化',
                                                      value=flags.Performance.has_restricted_features(modules.config.default_performance),
                                                      info='生成中の中間結果を非表示にし、最終ギャラリーのみ表示します。')

                        disable_seed_increment = gr.Checkbox(label='シード自動増加を無効化',
                                                             info='生成枚数が 1 より多い場合の自動シード増加を無効にします。',
                                                             value=False)
                        read_wildcards_in_order = gr.Checkbox(label="ワイルドカードを順番に読む", value=False)

                        black_out_nsfw = gr.Checkbox(label='NSFW を黒塗り', value=modules.config.default_black_out_nsfw,
                                                     interactive=not modules.config.default_black_out_nsfw,
                                                     info='NSFW が検出された場合に黒い画像を表示します。')

                        black_out_nsfw.change(lambda x: gr.update(value=x, interactive=not x),
                                              inputs=black_out_nsfw, outputs=disable_preview, queue=False,
                                              show_progress=False)

                        if not args_manager.args.disable_image_log:
                            save_final_enhanced_image_only = gr.Checkbox(label='最終強化画像のみ保存',
                                                                         value=modules.config.default_save_only_final_enhanced_image)

                        if not args_manager.args.disable_metadata:
                            save_metadata_to_images = gr.Checkbox(label='メタデータを画像に保存', value=modules.config.default_save_metadata_to_images,
                                                                  info='生成パラメータを画像に埋め込み、後から再生成できるようにします。')
                            metadata_scheme = gr.Radio(label='メタデータスキーム', choices=flags.metadata_scheme, value=modules.config.default_metadata_scheme,
                                                       info='画像プロンプトのパラメータは含まれません。Civitai との互換性には png と a1111 を使用してください。',
                                                       visible=modules.config.default_save_metadata_to_images)

                            save_metadata_to_images.change(lambda x: gr.update(visible=x), inputs=[save_metadata_to_images], outputs=[metadata_scheme],
                                                           queue=False, show_progress=False)

                    with gr.Tab(label='コントロール'):
                        debugging_cn_preprocessor = gr.Checkbox(label='プリプロセッサーをデバッグ', value=False,
                                                                info='プリプロセッサーの処理結果を確認します。')
                        skipping_cn_preprocessor = gr.Checkbox(label='プリプロセッサーをスキップ', value=False,
                                                               info='画像を前処理しません（入力が既に Canny/Depth 等の場合に使用）。')

                        mixing_image_prompt_and_vary_upscale = gr.Checkbox(label='画像プロンプトとバリエーション/拡大を混合',
                                                                           value=False)
                        mixing_image_prompt_and_inpaint = gr.Checkbox(label='画像プロンプトとインペイントを混合',
                                                                      value=False)

                        controlnet_softness = gr.Slider(label='ControlNet ソフトネス', minimum=0.0, maximum=1.0,
                                                        step=0.001, value=0.25,
                                                        info='A1111 のコントロールモードに相当します（0.0 で無効）。')

                        with gr.Tab(label='Canny'):
                            canny_low_threshold = gr.Slider(label='Canny 低閾値', minimum=1, maximum=255,
                                                            step=1, value=64)
                            canny_high_threshold = gr.Slider(label='Canny 高閾値', minimum=1, maximum=255,
                                                             step=1, value=128)

                    with gr.Tab(label='インペイント'):
                        debugging_inpaint_preprocessor = gr.Checkbox(label='インペイント前処理をデバッグ', value=False)
                        debugging_enhance_masks_checkbox = gr.Checkbox(label='強化マスクをデバッグ', value=False,
                                                                       info='プレビューと最終結果に強化マスクを表示します。')
                        debugging_dino = gr.Checkbox(label='GroundingDINO をデバッグ', value=False,
                                                     info='SAM マスクの代わりに GroundingDINO のボックスを使用します。')
                        inpaint_disable_initial_latent = gr.Checkbox(label='インペイントの初期潜在を無効化', value=False)
                        inpaint_engine = gr.Dropdown(label='インペイントエンジン',
                                                     value=modules.config.default_inpaint_engine_version,
                                                     choices=flags.inpaint_engine_versions,
                                                     info='Fooocus インペイントモデルのバージョン。Quality または Speed を推奨。')
                        inpaint_strength = gr.Slider(label='インペイント デノイズ強度',
                                                     minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                                                     info='A1111 インペイントのデノイズ強度と同じです。'
                                                          'インペイントのみ有効（アウトペイントは常に 1.0）。')
                        inpaint_respective_field = gr.Slider(label='インペイント対象範囲',
                                                             minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                                                             info='インペイントする領域。'
                                                                  '0 = A1111 の「マスクのみ」、'
                                                                  '1 = 「画像全体」。'
                                                                  'インペイントのみ有効（アウトペイントは常に 1.0）。')
                        inpaint_erode_or_dilate = gr.Slider(label='マスク 侵食/膨張',
                                                            minimum=-64, maximum=64, step=1, value=0,
                                                            info='正の値はマスクの白い領域を拡大、'
                                                                 '負の値は縮小します。'
                                                                 '（デフォルト 0、マスク反転前に処理）')
                        dino_erode_or_dilate = gr.Slider(label='GroundingDINO ボックス 侵食/膨張',
                                                         minimum=-64, maximum=64, step=1, value=0,
                                                         info='正の値はマスクの白い領域を拡大、'
                                                              '負の値は縮小します。'
                                                              '（デフォルト 0、SAM 処理前に実行）')

                        inpaint_mask_color = gr.ColorPicker(label='インペイントブラシの色', value='#FFFFFF', elem_id='inpaint_brush_color')

                        inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                                         inpaint_strength, inpaint_respective_field,
                                         inpaint_advanced_masking_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

                        inpaint_advanced_masking_checkbox.change(lambda x: [gr.update(visible=x)] * 2,
                                                                 inputs=inpaint_advanced_masking_checkbox,
                                                                 outputs=[inpaint_mask_image, inpaint_mask_generation_col],
                                                                 queue=False, show_progress=False)

                        inpaint_mask_color.change(lambda x: gr.update(brush_color=x), inputs=inpaint_mask_color,
                                                  outputs=inpaint_input_image,
                                                  queue=False, show_progress=False)

                    with gr.Tab(label='FreeU'):
                        freeu_enabled = gr.Checkbox(label='有効化', value=False)
                        freeu_b1 = gr.Slider(label='B1', minimum=0, maximum=2, step=0.01, value=1.01)
                        freeu_b2 = gr.Slider(label='B2', minimum=0, maximum=2, step=0.01, value=1.02)
                        freeu_s1 = gr.Slider(label='S1', minimum=0, maximum=4, step=0.01, value=0.99)
                        freeu_s2 = gr.Slider(label='S2', minimum=0, maximum=4, step=0.01, value=0.95)
                        freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

                def dev_mode_checked(r):
                    return gr.update(visible=r)

                dev_mode.change(dev_mode_checked, inputs=[dev_mode], outputs=[dev_tools],
                                queue=False, show_progress=False)

                def refresh_files_clicked():
                    modules.config.update_files()
                    results = [gr.update(choices=modules.config.model_filenames)]
                    results += [gr.update(choices=['None'] + modules.config.model_filenames)]
                    results += [gr.update(choices=[flags.default_vae] + modules.config.vae_filenames)]
                    if not args_manager.args.disable_preset_selection:
                        results += [gr.update(choices=modules.config.available_presets)]
                    for i in range(modules.config.default_max_lora_number):
                        results += [gr.update(interactive=True),
                                    gr.update(choices=['None'] + modules.config.lora_filenames), gr.update()]
                    return results

                refresh_files_output = [base_model, refiner_model, vae_name]
                if not args_manager.args.disable_preset_selection:
                    refresh_files_output += [preset_selection]
                refresh_files.click(refresh_files_clicked, [], refresh_files_output + lora_ctrls,
                                    queue=False, show_progress=False)

        state_is_generating = gr.State(False)

        load_data_outputs = [advanced_checkbox, image_number, prompt, negative_prompt, style_selections,
                             performance_selection, overwrite_step, overwrite_switch, aspect_ratios_selection,
                             overwrite_width, overwrite_height, guidance_scale, sharpness, adm_scaler_positive,
                             adm_scaler_negative, adm_scaler_end, refiner_swap_method, adaptive_cfg, clip_skip,
                             base_model, refiner_model, refiner_switch, sampler_name, scheduler_name, vae_name,
                             seed_random, image_seed, inpaint_engine, inpaint_engine_state,
                             inpaint_mode] + enhance_inpaint_mode_ctrls + [generate_button,
                             load_parameter_button] + freeu_ctrls + lora_ctrls

        if not args_manager.args.disable_preset_selection:
            def preset_selection_change(preset, is_generating, inpaint_mode):
                preset_content = modules.config.try_get_preset_content(preset) if preset != 'initial' else {}
                preset_prepared = modules.meta_parser.parse_meta_from_preset(preset_content)

                default_model = preset_prepared.get('base_model')
                previous_default_models = preset_prepared.get('previous_default_models', [])
                checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
                embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
                lora_downloads = preset_prepared.get('lora_downloads', {})
                vae_downloads = preset_prepared.get('vae_downloads', {})

                preset_prepared['base_model'], preset_prepared['checkpoint_downloads'] = launch.download_models(
                    default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads,
                    vae_downloads)

                if 'prompt' in preset_prepared and preset_prepared.get('prompt') == '':
                    del preset_prepared['prompt']

                return modules.meta_parser.load_parameter_button_click(json.dumps(preset_prepared), is_generating, inpaint_mode)


            def inpaint_engine_state_change(inpaint_engine_version, *args):
                if inpaint_engine_version == 'empty':
                    inpaint_engine_version = modules.config.default_inpaint_engine_version

                result = []
                for inpaint_mode in args:
                    if inpaint_mode != modules.flags.inpaint_option_detail:
                        result.append(gr.update(value=inpaint_engine_version))
                    else:
                        result.append(gr.update())

                return result

            preset_selection.change(preset_selection_change, inputs=[preset_selection, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}') \
                .then(inpaint_engine_state_change, inputs=[inpaint_engine_state] + enhance_inpaint_mode_ctrls, outputs=enhance_inpaint_engine_ctrls, queue=False, show_progress=False)

        # ── 保存先変更 イベント ──
        def _do_change_output_path(new_path):
            try:
                p = str(new_path).strip()
                if not p:
                    return gr.update(), '❌ パスを入力してください'
                os.makedirs(p, exist_ok=True)
                modules.config.path_outputs = p
                cfg = _ui_cfg_load()
                cfg['path_outputs'] = p
                _ui_cfg_save(cfg)
                return gr.update(value=p), f'✅ 保存先を変更しました: `{p}`'
            except Exception as e:
                return gr.update(), f'❌ エラー: {e}'

        output_path_btn.click(
            fn=_do_change_output_path, inputs=[output_path_box],
            outputs=[output_path_box, output_path_status], queue=False, show_progress=False)

        # ── User Setting Presets イベント ──
        _sp_save_inputs = [
            user_sp_name, prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number,
            guidance_scale, sharpness, base_model, refiner_model, refiner_switch,
            sampler_name, scheduler_name, vae_name,
            seed_random, image_seed,
            adm_scaler_positive, adm_scaler_negative, adm_scaler_end,
            refiner_swap_method, adaptive_cfg, clip_skip,
            input_image_checkbox, enhance_checkbox, uov_method,
        ] + freeu_ctrls + lora_ctrls

        def sp_do_save(name, prompt_v, neg_v, styles_v, perf_v, aspect_v, img_num,
                       guidance_v, sharp_v, base_v, ref_v, ref_sw_v,
                       sampler_v, sched_v, vae_v,
                       seed_rnd, seed_v,
                       adm_p, adm_n, adm_e,
                       ref_swap_v, adap_v, clip_v,
                       input_img_cb, enhance_cb, uov_m,
                       *rest):
            try:
                if not name or not str(name).strip():
                    return gr.update(), '❌ プリセット名を入力してください'
                n = str(name).strip()
                # rest = [freeu_en, freeu_b1, freeu_b2, freeu_s1, freeu_s2, lora_en1, lora_m1, lora_w1, ...]
                freeu_en = rest[0] if len(rest) > 0 else False
                freeu_b1_v = rest[1] if len(rest) > 1 else 1.01
                freeu_b2_v = rest[2] if len(rest) > 2 else 1.02
                freeu_s1_v = rest[3] if len(rest) > 3 else 0.99
                freeu_s2_v = rest[4] if len(rest) > 4 else 0.95
                lora_vals = rest[5:]

                meta = {
                    'prompt': str(prompt_v) if prompt_v else '',
                    'negative_prompt': str(neg_v) if neg_v else '',
                    'styles': str(styles_v) if styles_v else '[]',
                    'performance': str(perf_v) if perf_v else 'Speed',
                    'resolution': _sp_aspect_to_resolution(str(aspect_v) if aspect_v else ''),
                    'image_number': int(float(img_num)) if img_num is not None else 1,
                    'guidance_scale': float(guidance_v) if guidance_v is not None else 7.0,
                    'sharpness': float(sharp_v) if sharp_v is not None else 2.0,
                    'base_model': str(base_v) if base_v else '',
                    'refiner_model': str(ref_v) if ref_v else 'None',
                    'refiner_switch': float(ref_sw_v) if ref_sw_v is not None else 0.5,
                    'sampler': str(sampler_v) if sampler_v else 'dpmpp_2m_sde_gpu',
                    'scheduler': str(sched_v) if sched_v else 'karras',
                    'vae': str(vae_v) if vae_v else 'Default (model)',
                    'adm_guidance': f'({adm_p}, {adm_n}, {adm_e})',
                    'refiner_swap_method': str(ref_swap_v) if ref_swap_v else 'joint',
                    'adaptive_cfg': float(adap_v) if adap_v is not None else 7.0,
                    'clip_skip': int(float(clip_v)) if clip_v is not None else 2,
                    'input_image_enabled': bool(input_img_cb),
                    'enhance_enabled':     bool(enhance_cb),
                    'uov_method':          str(uov_m) if uov_m else flags.disabled,
                }
                if not seed_rnd:
                    try:
                        meta['seed'] = int(float(seed_v))
                    except Exception:
                        pass
                if freeu_en:
                    meta['freeu'] = f'({freeu_b1_v}, {freeu_b2_v}, {freeu_s1_v}, {freeu_s2_v})'
                for i in range(0, len(lora_vals), 3):
                    en = lora_vals[i]
                    nm = lora_vals[i + 1]
                    wt = lora_vals[i + 2]
                    meta[f'lora_combined_{i // 3 + 1}'] = f'{"True" if en else "False"} : {nm} : {wt}'

                presets = _sp_load()
                presets[n] = meta
                _sp_save(presets)
                choices = list(presets.keys())
                print(f'[UserPreset] 保存完了: {n} / 合計{len(choices)}件 → {_USER_SP_PATH}')
                return gr.update(choices=choices, value=n), f'✅ 保存しました：{n}'
            except Exception as e:
                import traceback
                msg = traceback.format_exc()
                print(f'[UserPreset] 保存エラー:\n{msg}')
                return gr.update(), f'❌ 保存エラー: {e}'

        def sp_do_load(name, is_gen, inpaint_m):
            _no_change = [gr.update()] * 5  # input_img_cb, enhance_cb, img_panel, enhance_panel, uov_m
            try:
                if not name:
                    return [gr.update()] * len(load_data_outputs) + ['❌ プリセットを選択してください'] + _no_change
                presets = _sp_load()
                if name not in presets:
                    return [gr.update()] * len(load_data_outputs) + [f'❌ 「{name}」が見つかりません'] + _no_change
                meta = presets[name]
                result = modules.meta_parser.load_parameter_button_click(meta, is_gen, inpaint_m)
                print(f'[UserPreset] 読込完了: {name}')
                # 追加フィールド: 入力画像 / 強化 / UoV メソッド
                img_en  = bool(meta.get('input_image_enabled', False))
                enh_en  = bool(meta.get('enhance_enabled', False))
                uov_val = meta.get('uov_method', flags.disabled)
                return (result
                        + [f'✅ 読み込みました：{name}']
                        + [gr.update(value=img_en),
                           gr.update(value=enh_en),
                           gr.update(visible=img_en),
                           gr.update(visible=enh_en),
                           gr.update(value=uov_val)])
            except Exception as e:
                import traceback
                msg = traceback.format_exc()
                print(f'[UserPreset] 読込エラー:\n{msg}')
                return [gr.update()] * len(load_data_outputs) + [f'❌ 読込エラー: {e}'] + _no_change

        def sp_do_delete(name):
            try:
                if not name:
                    return gr.update(), '❌ プリセットを選択してください'
                presets = _sp_load()
                if name not in presets:
                    return gr.update(), f'❌ 「{name}」が見つかりません'
                del presets[name]
                _sp_save(presets)
                choices = list(presets.keys())
                return gr.update(choices=choices, value=choices[0] if choices else None), f'🗑️ 削除しました：{name}'
            except Exception as e:
                return gr.update(), f'❌ 削除エラー: {e}'

        def sp_refresh_dropdown():
            choices = list(_sp_load().keys())
            return gr.update(choices=choices, value=choices[0] if choices else None)

        shared.gradio_root.load(fn=sp_refresh_dropdown, inputs=[], outputs=[user_sp_dropdown],
                                queue=False, show_progress=False)

        user_sp_save_btn.click(
            fn=sp_do_save,
            inputs=_sp_save_inputs,
            outputs=[user_sp_dropdown, user_sp_status],
            queue=False, show_progress=False)

        user_sp_load_btn.click(
            fn=sp_do_load,
            inputs=[user_sp_dropdown, state_is_generating, inpaint_mode],
            outputs=load_data_outputs + [user_sp_status,
                                         input_image_checkbox, enhance_checkbox,
                                         image_input_panel, enhance_input_panel, uov_method],
            queue=False, show_progress=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, _js='()=>{refresh_style_localization();}')

        user_sp_delete_btn.click(
            fn=sp_do_delete,
            inputs=[user_sp_dropdown],
            outputs=[user_sp_dropdown, user_sp_status],
            queue=False, show_progress=False)

        # ── ブラッシュアップ & OpenCV エディット イベント ──
        _brushup_outputs = [
            uov_input_image, uov_method, input_image_checkbox, image_input_panel,
            image_input_tabs, current_tab,
        ]
        _brushup_select_outputs = [
            brushup_selected, brushup_original, brushup_status_html,
            brushup_btn_col, edit_panel, edit_preview,
        ]

        def on_brushup_select(evt: gr.SelectData, paths_state):
            # Gradio 3.41.2: evt.value = caption (usually None), evt.index = int index
            _hide = gr.update(visible=False)
            _fail = (None, None,
                     '<span style="color:#5a5a7a;font-size:11px;font-family:monospace;">'
                     '💡 ギャラリーの画像をクリックして選択</span>',
                     _hide, _hide, None)
            try:
                idx = evt.index
                if not (paths_state and isinstance(idx, int) and 0 <= idx < len(paths_state)):
                    return _fail
                path = paths_state[idx]
                if not (isinstance(path, str) and os.path.exists(path)):
                    return _fail
                from PIL import Image as PILImage
                import numpy as np
                from modules.image_editor import resize_for_preview as _rfp
                arr = np.array(PILImage.open(path).convert('RGB'))
                h, w = arr.shape[:2]
                fname = os.path.basename(path)
                html = (f'<span style="color:#7fffb0;font-size:11px;font-family:monospace;">'
                        f'✓ {fname} ({w}×{h}px) — 方法を選択 または ↓ で編集</span>')
                preview = _rfp(arr, 768)
                return arr, arr, html, gr.update(visible=True), gr.update(visible=True), preview
            except Exception as e:
                return (None, None,
                        f'<span style="color:#ff7c7c;font-size:11px;">❌ {e}</span>',
                        _hide, _hide, None)

        gallery.select(fn=on_brushup_select, inputs=[gallery_paths], outputs=_brushup_select_outputs, queue=False, show_progress=False)
        progress_gallery.select(fn=on_brushup_select, inputs=[gallery_paths], outputs=_brushup_select_outputs, queue=False, show_progress=False)

        def do_brushup(img, method):
            if img is None:
                return (gr.update(), gr.update(), gr.update(), gr.update(visible=False),
                        gr.update(), gr.update())
            return (img, method, True, gr.update(visible=True),
                    gr.update(selected='uov_tab'), 'uov')

        _scroll_js = '() => { setTimeout(function(){ var el=document.getElementById("image_input_panel"); if(!el) el=document.querySelector(".image_input_panel"); if(el) el.scrollIntoView({behavior:"smooth",block:"start"}); }, 200); return []; }'

        brushup_subtle_btn.click(
            fn=lambda img: do_brushup(img, flags.subtle_variation),
            inputs=[brushup_selected], outputs=_brushup_outputs, queue=False, show_progress=False
        ).then(fn=lambda: None, _js=_scroll_js)

        brushup_strong_btn.click(
            fn=lambda img: do_brushup(img, flags.strong_variation),
            inputs=[brushup_selected], outputs=_brushup_outputs, queue=False, show_progress=False
        ).then(fn=lambda: None, _js=_scroll_js)

        brushup_upscale_btn.click(
            fn=lambda img: do_brushup(img, flags.upscale_2),
            inputs=[brushup_selected], outputs=_brushup_outputs, queue=False, show_progress=False
        ).then(fn=lambda: None, _js=_scroll_js)

        brushup_gen_btn.click(
            fn=lambda img: do_brushup(img, flags.subtle_variation),
            inputs=[brushup_selected], outputs=_brushup_outputs, queue=False, show_progress=False
        ).then(fn=lambda: None, _js='() => { setTimeout(function(){ var b=document.getElementById("generate_button"); if(b&&!b.disabled)b.click(); },600); return []; }')

        # ── OpenCV エディター イベント ──
        from modules.image_editor import apply_adjustments, resize_for_preview, rotate_image, flip_image

        # プリセット定義: (brightness, contrast, saturation, hue, temperature, sharpness)
        _PRESETS = {
            'vivid':  (10,  1.25, 1.6,  0,  0,   1.3),
            'cinema': (-8,  1.35, 0.7,  0,  15,  1.0),
            'warm':   (12,  1.05, 1.2,  0,  40,  1.0),
            'cool':   (0,   1.1,  0.85, 0, -40,  1.0),
            'soft':   (8,   0.88, 0.88, 0,  0,   0.4),
        }

        # スライダーの入力リスト (順番をそろえる)
        _edit_inputs = [brushup_original,
                        edit_brightness, edit_contrast, edit_saturation,
                        edit_hue, edit_temperature, edit_sharpness]
        _slider_outputs = [edit_brightness, edit_contrast, edit_saturation,
                           edit_hue, edit_temperature, edit_sharpness]

        def _update_preview(orig, brightness, contrast, saturation, hue_shift, temperature, sharpness):
            small = resize_for_preview(orig, 768)
            return apply_adjustments(small, brightness=brightness, contrast=contrast,
                                     saturation=saturation, hue_shift=hue_shift,
                                     sharpness=sharpness, temperature=temperature)

        # スライダー変更 → リアルタイムプレビュー
        for _sl in [edit_brightness, edit_contrast, edit_saturation, edit_hue, edit_temperature, edit_sharpness]:
            _sl.change(fn=_update_preview, inputs=_edit_inputs, outputs=edit_preview,
                       queue=False, show_progress=False)

        # リセット
        def _reset_edit(orig):
            preview = resize_for_preview(orig, 768)
            return 0, 1.0, 1.0, 0, 0, 1.0, preview

        edit_reset_btn.click(
            fn=_reset_edit, inputs=[brushup_original],
            outputs=_slider_outputs + [edit_preview],
            queue=False, show_progress=False
        )

        # 適用 → brushup_selected を編集版に更新
        def _apply_edit(orig, brightness, contrast, saturation, hue_shift, temperature, sharpness):
            edited = apply_adjustments(orig, brightness=brightness, contrast=contrast,
                                       saturation=saturation, hue_shift=hue_shift,
                                       sharpness=sharpness, temperature=temperature)
            if edited is None:
                return gr.update(), gr.update()
            preview = resize_for_preview(edited, 768)
            return edited, preview

        edit_apply_btn.click(
            fn=_apply_edit, inputs=_edit_inputs,
            outputs=[brushup_selected, edit_preview],
            queue=False, show_progress=False
        )

        # ── プリセット ──────────────────────────────
        def _apply_preset(key, orig):
            br, ct, sat, hue, temp, sharp = _PRESETS[key]
            preview = _update_preview(orig, br, ct, sat, hue, temp, sharp)
            return br, ct, sat, hue, temp, sharp, preview

        _preset_outs = _slider_outputs + [edit_preview]
        edit_preset_vivid.click( fn=lambda o: _apply_preset('vivid',  o), inputs=[brushup_original], outputs=_preset_outs, queue=False, show_progress=False)
        edit_preset_cinema.click(fn=lambda o: _apply_preset('cinema', o), inputs=[brushup_original], outputs=_preset_outs, queue=False, show_progress=False)
        edit_preset_warm.click(  fn=lambda o: _apply_preset('warm',   o), inputs=[brushup_original], outputs=_preset_outs, queue=False, show_progress=False)
        edit_preset_cool.click(  fn=lambda o: _apply_preset('cool',   o), inputs=[brushup_original], outputs=_preset_outs, queue=False, show_progress=False)
        edit_preset_soft.click(  fn=lambda o: _apply_preset('soft',   o), inputs=[brushup_original], outputs=_preset_outs, queue=False, show_progress=False)

        # ── 回転・反転 ──────────────────────────────
        _transform_inputs  = [brushup_original] + _slider_outputs
        _transform_outputs = [brushup_original, edit_preview]

        def _do_rotate(orig, brightness, contrast, saturation, hue_shift, temperature, sharpness, deg):
            rotated = rotate_image(orig, deg)
            if rotated is None:
                return gr.update(), gr.update()
            return rotated, _update_preview(rotated, brightness, contrast, saturation, hue_shift, temperature, sharpness)

        def _do_flip(orig, brightness, contrast, saturation, hue_shift, temperature, sharpness, direction):
            flipped = flip_image(orig, direction)
            if flipped is None:
                return gr.update(), gr.update()
            return flipped, _update_preview(flipped, brightness, contrast, saturation, hue_shift, temperature, sharpness)

        edit_rot_l.click( fn=lambda *a: _do_rotate(*a, 270), inputs=_transform_inputs, outputs=_transform_outputs, queue=False, show_progress=False)
        edit_rot_r.click( fn=lambda *a: _do_rotate(*a,  90), inputs=_transform_inputs, outputs=_transform_outputs, queue=False, show_progress=False)
        edit_flip_h.click(fn=lambda *a: _do_flip(*a,   'h'), inputs=_transform_inputs, outputs=_transform_outputs, queue=False, show_progress=False)
        edit_flip_v.click(fn=lambda *a: _do_flip(*a,   'v'), inputs=_transform_inputs, outputs=_transform_outputs, queue=False, show_progress=False)

        # ── Before / After ─────────────────────────
        edit_ba_btn.click(
            fn=lambda orig: resize_for_preview(orig, 768),
            inputs=[brushup_original], outputs=edit_preview,
            queue=False, show_progress=False
        )

        performance_selection.change(lambda x: [gr.update(interactive=not flags.Performance.has_restricted_features(x))] * 11 +
                                               [gr.update(visible=not flags.Performance.has_restricted_features(x))] * 1 +
                                               [gr.update(value=flags.Performance.has_restricted_features(x))] * 1,
                                     inputs=performance_selection,
                                     outputs=[
                                         guidance_scale, sharpness, adm_scaler_end, adm_scaler_positive,
                                         adm_scaler_negative, refiner_switch, refiner_model, sampler_name,
                                         scheduler_name, adaptive_cfg, refiner_swap_method, negative_prompt, disable_intermediate_results
                                     ], queue=False, show_progress=False)

        output_format.input(lambda x: gr.update(output_format=x), inputs=output_format)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, advanced_column,
                                 queue=False, show_progress=False) \
            .then(fn=lambda: None, _js='refresh_grid_delayed', queue=False, show_progress=False)

        inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[
            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,
            inpaint_disable_initial_latent, inpaint_engine,
            inpaint_strength, inpaint_respective_field
        ], show_progress=False, queue=False)

        # load configured default_inpaint_method
        default_inpaint_ctrls = [inpaint_mode, inpaint_disable_initial_latent, inpaint_engine, inpaint_strength, inpaint_respective_field]
        for mode, disable_initial_latent, engine, strength, respective_field in [default_inpaint_ctrls] + enhance_inpaint_update_ctrls:
            shared.gradio_root.load(inpaint_mode_change, inputs=[mode, inpaint_engine_state], outputs=[
                inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts, disable_initial_latent,
                engine, strength, respective_field
            ], show_progress=False, queue=False)

        generate_mask_button.click(fn=generate_mask,
                                   inputs=[inpaint_input_image, inpaint_mask_model, inpaint_mask_cloth_category,
                                           inpaint_mask_dino_prompt_text, inpaint_mask_sam_model,
                                           inpaint_mask_box_threshold, inpaint_mask_text_threshold,
                                           inpaint_mask_sam_max_detections, dino_erode_or_dilate, debugging_dino],
                                   outputs=inpaint_mask_image, show_progress=True, queue=True)

        ctrls = [currentTask, generate_image_grid]
        ctrls += [
            prompt, negative_prompt, style_selections,
            performance_selection, aspect_ratios_selection, image_number, output_format, image_seed,
            read_wildcards_in_order, sharpness, guidance_scale
        ]

        ctrls += [base_model, refiner_model, refiner_switch] + lora_ctrls
        ctrls += [input_image_checkbox, current_tab]
        ctrls += [uov_method, uov_input_image]
        ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
        ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment, black_out_nsfw]
        ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg, clip_skip]
        ctrls += [sampler_name, scheduler_name, vae_name]
        ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
        ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
        ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
        ctrls += [refiner_swap_method, controlnet_softness]
        ctrls += freeu_ctrls
        ctrls += inpaint_ctrls

        if not args_manager.args.disable_image_log:
            ctrls += [save_final_enhanced_image_only]

        if not args_manager.args.disable_metadata:
            ctrls += [save_metadata_to_images, metadata_scheme]

        ctrls += ip_ctrls
        ctrls += [debugging_dino, dino_erode_or_dilate, debugging_enhance_masks_checkbox,
                  enhance_input_image, enhance_checkbox, enhance_uov_method, enhance_uov_processing_order,
                  enhance_uov_prompt_type]
        ctrls += enhance_ctrls

        def parse_meta(raw_prompt_txt, is_generating):
            loaded_json = None
            if is_json(raw_prompt_txt):
                loaded_json = json.loads(raw_prompt_txt)

            if loaded_json is None:
                if is_generating:
                    return gr.update(), gr.update(), gr.update()
                else:
                    return gr.update(), gr.update(visible=True), gr.update(visible=False)

            return json.dumps(loaded_json), gr.update(visible=False), gr.update(visible=True)

        prompt.input(parse_meta, inputs=[prompt, state_is_generating], outputs=[prompt, generate_button, load_parameter_button], queue=False, show_progress=False)

        load_parameter_button.click(modules.meta_parser.load_parameter_button_click, inputs=[prompt, state_is_generating, inpaint_mode], outputs=load_data_outputs, queue=False, show_progress=False)

        def trigger_metadata_import(file, state_is_generating):
            parameters, metadata_scheme = modules.meta_parser.read_info_from_image(file)
            if parameters is None:
                print('画像にメタデータが見つかりませんでした。')
                parsed_parameters = {}
            else:
                metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                parsed_parameters = metadata_parser.to_json(parameters)

            return modules.meta_parser.load_parameter_button_click(parsed_parameters, state_is_generating, inpaint_mode)

        metadata_import_button.click(trigger_metadata_import, inputs=[metadata_input_image, state_is_generating], outputs=load_data_outputs, queue=False, show_progress=True) \
            .then(style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False)

        generate_button.click(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True),
                              outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating]) \
            .then(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed) \
            .then(fn=get_task, inputs=ctrls, outputs=currentTask) \
            .then(fn=generate_clicked, inputs=currentTask, outputs=[progress_html, progress_window, progress_gallery, gallery, gallery_paths]) \
            .then(lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False),
                  outputs=[generate_button, stop_button, skip_button, state_is_generating]) \
            .then(fn=update_history_link, outputs=history_link) \
            .then(fn=lambda: None, _js='playNotification').then(fn=lambda: None, _js='refresh_grid_delayed')

        reset_button.click(lambda: [worker.AsyncTask(args=[]), False, gr.update(visible=True, interactive=True)] +
                                   [gr.update(visible=False)] * 6 +
                                   [gr.update(visible=True, value=[])],
                           outputs=[currentTask, state_is_generating, generate_button,
                                    reset_button, stop_button, skip_button,
                                    progress_html, progress_window, progress_gallery, gallery],
                           queue=False)

        for notification_file in ['notification.ogg', 'notification.mp3']:
            if os.path.exists(notification_file):
                gr.Audio(interactive=False, value=notification_file, elem_id='audio_notification', visible=False)
                break

        def trigger_describe(modes, img, apply_styles):
            describe_prompts = []
            styles = set()

            if flags.describe_type_photo in modes:
                from extras.interrogate import default_interrogator as default_interrogator_photo
                describe_prompts.append(default_interrogator_photo(img))
                styles.update(["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"])

            if flags.describe_type_anime in modes:
                from extras.wd14tagger import default_interrogator as default_interrogator_anime
                describe_prompts.append(default_interrogator_anime(img))
                styles.update(["Fooocus V2", "Fooocus Masterpiece"])

            if len(styles) == 0 or not apply_styles:
                styles = gr.update()
            else:
                styles = list(styles)

            if len(describe_prompts) == 0:
                describe_prompt = gr.update()
            else:
                describe_prompt = ', '.join(describe_prompts)

            return describe_prompt, styles

        describe_btn.click(trigger_describe, inputs=[describe_methods, describe_input_image, describe_apply_styles],
                           outputs=[prompt, style_selections], show_progress=True, queue=True) \
            .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
            .then(lambda: None, _js='()=>{refresh_style_localization();}')

        # ── Prompt Forge 統合：postMessage リスナー ──
        shared.gradio_root.load(
            fn=lambda: None, inputs=[], outputs=[],
            queue=False, show_progress=False,
            _js="""
function() {
  if (window._pfListenerAdded) return [];
  window._pfListenerAdded = true;

  // React管理のtextareaに値をセットしてイベントを発火する
  function pfSetTextarea(selector, value) {
    var el = document.querySelector(selector);
    if (!el) return false;
    try {
      var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
      nativeSetter.call(el, value);
    } catch(e) {
      el.value = value;
    }
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
    return true;
  }

  // 転送後、テキストエリアをハイライト（緑フラッシュ）
  function pfFlash(selector) {
    var el = document.querySelector(selector);
    if (!el) return;
    el.style.transition = 'box-shadow 0.15s ease';
    el.style.boxShadow = '0 0 0 3px rgba(100,255,150,0.7)';
    setTimeout(function() { el.style.boxShadow = ''; }, 1000);
  }

  // アコーディオンを閉じる（複数のセレクタを試みる）
  function pfCloseAccordion() {
    var btn = document.querySelector('#pf_accordion button[aria-expanded="true"]');
    if (!btn) btn = document.querySelector('#pf_accordion .toggle-icon[aria-expanded="true"]');
    if (!btn) btn = document.querySelector('#pf_accordion > button:first-child');
    if (btn) {
      try { btn.click(); } catch(e) {}
    }
  }

  window.addEventListener('message', function(ev) {
    if (!ev.data || ev.data.type !== 'pf_send') return;

    // アコーディオンを閉じる
    pfCloseAccordion();

    // positive_prompt を更新
    var posOk = pfSetTextarea('#positive_prompt textarea', ev.data.pos || '');

    // negative_prompt を更新（Settingsタブが閉じていてもGradioはDOMに保持する）
    pfSetTextarea('#negative_prompt textarea', ev.data.neg || '');

    // フォーカス＆フラッシュ
    setTimeout(function() {
      var posEl = document.querySelector('#positive_prompt textarea');
      if (posEl) {
        posEl.focus();
        posEl.scrollIntoView({behavior: 'smooth', block: 'center'});
      }
      pfFlash('#positive_prompt textarea');
    }, 150);

    // 生成ボタン自動クリック
    if (ev.data.generate) {
      setTimeout(function() {
        var btn = document.getElementById('generate_button');
        if (btn) btn.click();
      }, 600);
    }
  });
  return [];
}
"""
        )

        if args_manager.args.enable_auto_describe_image:
            def trigger_auto_describe(mode, img, prompt, apply_styles):
                # keep prompt if not empty
                if prompt == '':
                    return trigger_describe(mode, img, apply_styles)
                return gr.update(), gr.update()

            uov_input_image.upload(trigger_auto_describe, inputs=[describe_methods, uov_input_image, prompt, describe_apply_styles],
                                   outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

            enhance_input_image.upload(lambda: gr.update(value=True), outputs=enhance_checkbox, queue=False, show_progress=False) \
                .then(trigger_auto_describe, inputs=[describe_methods, enhance_input_image, prompt, describe_apply_styles],
                      outputs=[prompt, style_selections], show_progress=True, queue=True) \
                .then(fn=style_sorter.sort_styles, inputs=style_selections, outputs=style_selections, queue=False, show_progress=False) \
                .then(lambda: None, _js='()=>{refresh_style_localization();}')

def dump_default_english_config():
    from modules.localization import dump_english_config
    dump_english_config(grh.all_components)


# dump_default_english_config()

# ── Prompt Forge ブリッジサーバーを起動 ──
try:
    import prompt_forge_bridge
    prompt_forge_bridge.start_bridge()
except Exception as _pf_err:
    print(f'[PromptForge] ブリッジ起動失敗: {_pf_err}')

shared.gradio_root.launch(
    inbrowser=args_manager.args.in_browser,
    server_name=args_manager.args.listen,
    server_port=args_manager.args.port,
    share=args_manager.args.share,
    auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
    allowed_paths=[modules.config.path_outputs],
    blocked_paths=[constants.AUTH_FILENAME]
)
