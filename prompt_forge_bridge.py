"""
Prompt Forge Bridge Server
--------------------------
Fooocusプロセス内でバックグラウンドスレッドとして動作するHTTPサーバー。
- ポート8080でPrompt ForgeのHTMLを配信
- /v1/generation/text-to-image  REST APIを提供
- Fooocusのworkerキューに直接AsyncTaskを投入して画像生成
"""

import json
import os
import base64
import threading
import time
import io
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

BRIDGE_PORT = 8080
_html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompt_forge_v3_11.html')


# ──────────────────────────────────────────────────────────────────
# AsyncTask args ビルダー
# ──────────────────────────────────────────────────────────────────
def _build_args(req: dict) -> list:
    """リクエストJSONからAsyncTask argsリストを構築する。"""
    import modules.config as config
    import modules.flags as flags
    import args_manager

    # ── ユーザー指定パラメータ ──
    prompt          = req.get('prompt', '')
    negative_prompt = req.get('negative_prompt', '')
    styles          = req.get('style_selections', list(config.default_styles))
    if isinstance(styles, str):
        styles = [styles]
    performance     = req.get('performance_selection', 'Speed')
    image_number    = int(req.get('image_number', 1))
    output_format   = req.get('output_format', 'png')
    image_seed      = int(req.get('image_seed', -1))
    sharpness       = float(req.get('sharpness', config.default_sample_sharpness))
    guidance_scale  = float(req.get('guidance_scale', config.default_cfg_scale))

    # アスペクト比を Fooocus 形式に変換 ("1024*1024" → "1024×1024 | 1:1")
    aspect_raw = str(req.get('aspect_ratios_selection', '1024*1024'))
    aspect_raw = aspect_raw.replace('×', '*').replace('x', '*').replace('X', '*')
    if '*' not in aspect_raw:
        aspect_raw = '1024*1024'
    aspect_sel = config.add_ratio(aspect_raw)

    args = [
        False,           # generate_image_grid
        prompt,
        negative_prompt,
        styles,
        performance,
        aspect_sel,
        image_number,
        output_format,
        image_seed,
        False,           # read_wildcards_in_order
        sharpness,
        guidance_scale,
        config.default_base_model_name,
        config.default_refiner_model_name,
        config.default_refiner_switch,
    ]

    # LoRA (default_max_lora_number x 3: enabled, name, weight)
    # default_loras は必ず default_max_lora_number 個にパディングされている
    loras_to_add = config.default_loras[:config.default_max_lora_number]
    # 不足分を 'None' で補填
    while len(loras_to_add) < config.default_max_lora_number:
        loras_to_add.append([True, 'None', 1.0])
    for lora in loras_to_add:
        args.extend([bool(lora[0]), str(lora[1]), float(lora[2])])

    args += [
        False,           # input_image_checkbox
        'uov',           # current_tab
        flags.disabled,  # uov_method ('Disabled')
        None,            # uov_input_image
        [],              # outpaint_selections
        {'image': None, 'mask': None},  # inpaint_input_image
        '',              # inpaint_additional_prompt
        None,            # inpaint_mask_image_upload
        False,           # disable_preview
        False,           # disable_intermediate_results
        False,           # disable_seed_increment
        config.default_black_out_nsfw,
        1.5,             # adm_scaler_positive
        0.8,             # adm_scaler_negative
        0.3,             # adm_scaler_end
        config.default_cfg_tsnr,   # adaptive_cfg
        config.default_clip_skip,  # clip_skip
        config.default_sampler,    # sampler_name
        config.default_scheduler,  # scheduler_name
        flags.default_vae,         # vae_name
        -1,              # overwrite_step
        -1,              # overwrite_switch
        -1,              # overwrite_width
        -1,              # overwrite_height
        -1,              # overwrite_vary_strength
        -1,              # overwrite_upscale_strength
        False,           # mixing_image_prompt_and_vary_upscale
        False,           # mixing_image_prompt_and_inpaint
        False,           # debugging_cn_preprocessor
        False,           # skipping_cn_preprocessor
        100,             # canny_low_threshold
        200,             # canny_high_threshold
        flags.refiner_swap_method,  # 'joint'
        0.25,            # controlnet_softness
        False,           # freeu_enabled
        1.01,            # freeu_b1
        1.02,            # freeu_b2
        0.99,            # freeu_s1
        0.95,            # freeu_s2
        # ── inpaint_ctrls ──
        False,           # debugging_inpaint_preprocessor
        False,           # inpaint_disable_initial_latent
        config.default_inpaint_engine_version,  # inpaint_engine
        1.0,             # inpaint_strength
        0.618,           # inpaint_respective_field
        False,           # inpaint_advanced_masking_checkbox
        False,           # invert_mask_checkbox
        0,               # inpaint_erode_or_dilate
    ]

    if not args_manager.args.disable_image_log:
        args.append(config.default_save_only_final_enhanced_image)
    if not args_manager.args.disable_metadata:
        args.append(config.default_save_metadata_to_images)
        args.append('fooocus')  # metadata_scheme

    # ControlNet IP (default_controlnet_image_count x 4)
    for _ in range(config.default_controlnet_image_count):
        args += [None, 0.5, 0.6, flags.default_ip]  # img, stop, weight, type

    # enhance 制御
    args += [
        False,  # debugging_dino
        0,      # dino_erode_or_dilate
        False,  # debugging_enhance_masks_checkbox
        None,   # enhance_input_image
        False,  # enhance_checkbox
        config.default_enhance_uov_method,
        config.default_enhance_uov_processing_order,
        config.default_enhance_uov_prompt_type,
    ]

    # enhance タブ (default_enhance_tabs x 16)
    for _ in range(config.default_enhance_tabs):
        args += [
            False,   # enhance_enabled
            '',      # enhance_mask_dino_prompt_text
            '',      # enhance_prompt
            '',      # enhance_negative_prompt
            config.default_enhance_inpaint_mask_model,   # 'sam'
            config.default_inpaint_mask_cloth_category,  # 'full'
            config.default_inpaint_mask_sam_model,       # 'vit_b'
            0.25,    # enhance_mask_text_threshold
            0.3,     # enhance_mask_box_threshold
            config.default_sam_max_detections,
            False,   # enhance_inpaint_disable_initial_latent
            config.default_inpaint_engine_version,
            0.5,     # enhance_inpaint_strength
            0.618,   # enhance_inpaint_respective_field
            0,       # enhance_inpaint_erode_or_dilate
            False,   # enhance_mask_invert
        ]

    print(f'[PromptForge] args構築完了: {len(args)}個 '
          f'(max_lora={config.default_max_lora_number}, '
          f'cn_count={config.default_controlnet_image_count}, '
          f'enhance_tabs={config.default_enhance_tabs})')
    return args


# ──────────────────────────────────────────────────────────────────
# 画像生成メイン
# ──────────────────────────────────────────────────────────────────
def _generate(req: dict) -> list:
    """
    Fooocusのworkerキューに生成タスクを投入し、完成した画像をbase64リストで返す。
    """
    import modules.async_worker as worker
    from modules.async_worker import AsyncTask

    args = _build_args(req)
    task = AsyncTask(args=args)
    worker.async_tasks.append(task)

    print(f'[PromptForge] 生成開始: "{task.prompt[:60]}"')

    timeout = 600  # 10分
    start = time.time()
    results_b64 = []

    while time.time() - start < timeout:
        time.sleep(0.3)

        if task.last_stop:
            raise RuntimeError('生成がキャンセルされました')

        # yields をすべて処理
        while task.yields:
            flag, product = task.yields.pop(0)
            if flag == 'finish':
                for img in product:
                    try:
                        if isinstance(img, str) and os.path.exists(img):
                            with open(img, 'rb') as f:
                                b64 = base64.b64encode(f.read()).decode('utf-8')
                        elif hasattr(img, 'save'):
                            buf = io.BytesIO()
                            img.save(buf, format='PNG')
                            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        else:
                            continue
                        results_b64.append({
                            'base64': b64,
                            'url': '',
                            'seed': task.seed if hasattr(task, 'seed') else req.get('image_seed', -1),
                            'finish_reason': 'SUCCESS',
                            'meta': {}
                        })
                    except Exception as e:
                        print(f'[PromptForge] 画像読込エラー: {e}')
                if results_b64:
                    print(f'[PromptForge] 完了: {len(results_b64)}枚')
                    return results_b64
                raise RuntimeError('画像データを取得できませんでした')

    raise TimeoutError('生成タイムアウト (600秒)')


# ──────────────────────────────────────────────────────────────────
# HTTP ハンドラ
# ──────────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # コンソールへの詳細ログを抑制

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Accept')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path.rstrip('/') or '/'
        if path in ('/', '/index.html'):
            if os.path.exists(_html_path):
                with open(_html_path, 'rb') as f:
                    data = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self._cors()
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'prompt_forge_v3_11.html not found')
        elif path == '/ping':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self._cors()
            self.end_headers()
            self.wfile.write(b'pong')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == '/v1/generation/text-to-image':
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                req = json.loads(body.decode('utf-8')) if body else {}
                images = _generate(req)
                resp = json.dumps(images).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self._cors()
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                import traceback
                msg = traceback.format_exc()
                print(f'[PromptForge] エラー:\n{msg}')
                resp = json.dumps({'error': str(e), 'traceback': msg}).encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self._cors()
                self.end_headers()
                self.wfile.write(resp)
        else:
            self.send_response(404)
            self.end_headers()


# ──────────────────────────────────────────────────────────────────
# 起動関数（webui.py から呼ばれる）
# ──────────────────────────────────────────────────────────────────
def start_bridge(port: int = BRIDGE_PORT):
    """ブリッジサーバーをデーモンスレッドで起動する。"""
    server = HTTPServer(('127.0.0.1', port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f'\n{"="*55}')
    print(f'  🎨 Prompt Forge が起動しました!')
    print(f'  ブラウザで以下を開いてください:')
    print(f'  http://127.0.0.1:{port}')
    print(f'{"="*55}\n')
    return server, t
