"""
FOOOCUS FORGE LAUNCHER
tkinter dark-themed GUI launcher for Fooocus + Prompt Forge Bridge
"""
import sys
import os
import threading
import subprocess
import socket
import time
import webbrowser
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

# ── カラーテーマ ──────────────────────────────────────────────────
BG       = "#0a0a0f"
BG2      = "#12121a"
BG3      = "#1a1a28"
BORDER   = "#2a2a3d"
FG       = "#e8e8f0"
FG2      = "#8888aa"
ACCENT   = "#7c4dff"
ACCENT2  = "#b388ff"
ACCENT3  = "#00e5ff"
GREEN    = "#00e676"
RED      = "#ff1744"
YELLOW   = "#ffea00"
ORANGE   = "#ff6d00"
X_BLUE   = "#1da1f2"

# ── ポート定義 ─────────────────────────────────────────────────────
PORT_FOOOCUS = 7865
PORT_BRIDGE  = 8080

# ── 起動コマンド ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _py():
    emb = os.path.join(BASE_DIR, "python_embedded", "python.exe")
    return emb if os.path.exists(emb) else sys.executable

def build_fooocus_cmd(preset="", in_browser=False, listen=False):
    cmd = [_py(), "launch.py"]
    if preset:
        cmd += ["--preset", preset]
    if in_browser:
        cmd.append("--in-browser")
    if listen:
        cmd.append("--listen")
    return cmd

def is_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


# ══════════════════════════════════════════════════════════════════
class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("FOOOCUS FORGE LAUNCHER")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(680, 520)

        # プロセス管理
        self._proc_fooocus: subprocess.Popen | None = None
        self._log_queue: queue.Queue = queue.Queue()

        # ── ウィジェット構築 ──
        self._build_header()
        self._build_status_bar()
        self._build_options()
        self._build_controls()
        self._build_log()
        self._build_links()

        # ── イベント ──
        self.bind("<Escape>", lambda _: self._on_close())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── ポーリング開始 ──
        self._poll_status()
        self._drain_log_queue()

    # ── ヘッダー ──────────────────────────────────────────────────
    def _build_header(self):
        f = tk.Frame(self, bg=BG, pady=12)
        f.pack(fill="x", padx=20)
        tk.Label(f, text="⚡ FOOOCUS FORGE LAUNCHER",
                 bg=BG, fg=ACCENT2,
                 font=("Consolas", 16, "bold")).pack(side="left")
        tk.Label(f, text="v1.0",
                 bg=BG, fg=FG2,
                 font=("Consolas", 10)).pack(side="left", padx=(8, 0), pady=4)

    # ── ステータスバー ────────────────────────────────────────────
    def _build_status_bar(self):
        outer = tk.Frame(self, bg=BG3, bd=0, highlightbackground=BORDER,
                         highlightthickness=1)
        outer.pack(fill="x", padx=20, pady=(0, 8))

        inner = tk.Frame(outer, bg=BG3, pady=8)
        inner.pack(fill="x", padx=12)

        # Fooocus
        fc = tk.Frame(inner, bg=BG3)
        fc.pack(side="left", padx=(0, 24))
        self._dot_fooocus = tk.Label(fc, text="●", fg=RED, bg=BG3,
                                     font=("Consolas", 14))
        self._dot_fooocus.pack(side="left")
        tk.Label(fc, text=f"  Fooocus  :{PORT_FOOOCUS}",
                 bg=BG3, fg=FG, font=("Consolas", 11)).pack(side="left")
        self._lbl_fooocus = tk.Label(fc, text="停止中",
                                     bg=BG3, fg=RED,
                                     font=("Consolas", 10))
        self._lbl_fooocus.pack(side="left", padx=(6, 0))

        sep = tk.Frame(inner, bg=BORDER, width=1, height=24)
        sep.pack(side="left", padx=12)

        # Bridge
        bc = tk.Frame(inner, bg=BG3)
        bc.pack(side="left")
        self._dot_bridge = tk.Label(bc, text="●", fg=RED, bg=BG3,
                                    font=("Consolas", 14))
        self._dot_bridge.pack(side="left")
        tk.Label(bc, text=f"  Bridge   :{PORT_BRIDGE}",
                 bg=BG3, fg=FG, font=("Consolas", 11)).pack(side="left")
        self._lbl_bridge = tk.Label(bc, text="停止中",
                                    bg=BG3, fg=RED,
                                    font=("Consolas", 10))
        self._lbl_bridge.pack(side="left", padx=(6, 0))

    # ── 起動オプション ────────────────────────────────────────────
    def _build_options(self):
        outer = tk.Frame(self, bg=BG2, bd=0, highlightbackground=BORDER,
                         highlightthickness=1)
        outer.pack(fill="x", padx=20, pady=(0, 8))

        inner = tk.Frame(outer, bg=BG2, pady=8, padx=12)
        inner.pack(fill="x")

        tk.Label(inner, text="起動オプション", bg=BG2, fg=FG2,
                 font=("Consolas", 9, "bold")).grid(
                     row=0, column=0, columnspan=6, sticky="w", pady=(0, 6))

        # プリセット
        tk.Label(inner, text="プリセット:", bg=BG2, fg=FG,
                 font=("Consolas", 10)).grid(row=1, column=0, sticky="w", padx=(0, 6))
        self._preset_var = tk.StringVar(value="なし")
        preset_cb = ttk.Combobox(inner, textvariable=self._preset_var,
                                  values=["なし", "anime", "realistic", "lcm"],
                                  state="readonly", width=12,
                                  font=("Consolas", 10))
        preset_cb.grid(row=1, column=1, sticky="w", padx=(0, 20))
        self._style_combobox(preset_cb)

        # ブラウザ自動起動
        self._browser_var = tk.BooleanVar(value=False)
        chk_browser = tk.Checkbutton(inner, text="ブラウザ自動起動",
                                      variable=self._browser_var,
                                      bg=BG2, fg=FG, selectcolor=BG3,
                                      activebackground=BG2, activeforeground=ACCENT2,
                                      font=("Consolas", 10))
        chk_browser.grid(row=1, column=2, sticky="w", padx=(0, 20))

        # --listen
        self._listen_var = tk.BooleanVar(value=False)
        chk_listen = tk.Checkbutton(inner, text="--listen (LAN公開)",
                                     variable=self._listen_var,
                                     bg=BG2, fg=FG, selectcolor=BG3,
                                     activebackground=BG2, activeforeground=ACCENT2,
                                     font=("Consolas", 10))
        chk_listen.grid(row=1, column=3, sticky="w")

    # ── コントロールボタン ────────────────────────────────────────
    def _build_controls(self):
        f = tk.Frame(self, bg=BG, pady=4)
        f.pack(fill="x", padx=20)

        btn_cfg = dict(font=("Consolas", 11, "bold"), cursor="hand2",
                       relief="flat", bd=0, padx=18, pady=7)

        self._btn_start = tk.Button(
            f, text="▶  起動", bg=ACCENT, fg="white",
            activebackground=ACCENT2, activeforeground="white",
            command=self._start_all, **btn_cfg)
        self._btn_start.pack(side="left", padx=(0, 8))

        self._btn_stop = tk.Button(
            f, text="■  停止", bg=BG3, fg=RED,
            activebackground=BORDER, activeforeground=RED,
            command=self._stop_all, state="disabled", **btn_cfg)
        self._btn_stop.pack(side="left", padx=(0, 8))

        self._btn_restart = tk.Button(
            f, text="↺  再起動", bg=BG3, fg=YELLOW,
            activebackground=BORDER, activeforeground=YELLOW,
            command=self._restart_all, state="disabled", **btn_cfg)
        self._btn_restart.pack(side="left", padx=(0, 8))

        tk.Button(f, text="ログクリア", bg=BG3, fg=FG2,
                  activebackground=BORDER, activeforeground=FG,
                  command=self._clear_log, **btn_cfg).pack(side="right")

    # ── ログ ──────────────────────────────────────────────────────
    def _build_log(self):
        f = tk.Frame(self, bg=BG)
        f.pack(fill="both", expand=True, padx=20, pady=(8, 4))

        tk.Label(f, text="ログ", bg=BG, fg=FG2,
                 font=("Consolas", 9, "bold")).pack(anchor="w", pady=(0, 3))

        self._log = scrolledtext.ScrolledText(
            f, bg=BG2, fg=FG, insertbackground=FG,
            font=("Consolas", 9), relief="flat",
            bd=0, highlightbackground=BORDER, highlightthickness=1,
            state="disabled", wrap="word")
        self._log.pack(fill="both", expand=True)

        # タグ色
        self._log.tag_config("info",    foreground=FG2)
        self._log.tag_config("ok",      foreground=GREEN)
        self._log.tag_config("warn",    foreground=YELLOW)
        self._log.tag_config("err",     foreground=RED)
        self._log.tag_config("accent",  foreground=ACCENT3)

    # ── クイックリンク ────────────────────────────────────────────
    def _build_links(self):
        f = tk.Frame(self, bg=BG, pady=8)
        f.pack(fill="x", padx=20)

        link_cfg = dict(font=("Consolas", 10), cursor="hand2",
                        relief="flat", bd=0, padx=12, pady=5)

        tk.Button(f, text="🌐 Prompt Forge  http://127.0.0.1:8080",
                  bg=BG3, fg=ACCENT3,
                  activebackground=BORDER, activeforeground=ACCENT3,
                  command=lambda: webbrowser.open("http://127.0.0.1:8080"),
                  **link_cfg).pack(side="left", padx=(0, 8))

        tk.Button(f, text="🌐 Fooocus UI  http://127.0.0.1:7865",
                  bg=BG3, fg=ACCENT2,
                  activebackground=BORDER, activeforeground=ACCENT2,
                  command=lambda: webbrowser.open("http://127.0.0.1:7865"),
                  **link_cfg).pack(side="left")

    # ── Combobox スタイル ─────────────────────────────────────────
    def _style_combobox(self, cb):
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TCombobox",
                         fieldbackground=BG3,
                         background=BG3,
                         foreground=FG,
                         selectbackground=ACCENT,
                         selectforeground="white",
                         arrowcolor=FG2)
        style.map("TCombobox",
                   fieldbackground=[("readonly", BG3)],
                   foreground=[("readonly", FG)],
                   background=[("readonly", BG3)])
        self.option_add("*TCombobox*Listbox.background", BG3)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.font", ("Consolas", 10))

    # ── ログ書き込み ──────────────────────────────────────────────
    def _log_write(self, text: str, tag: str = "info"):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    # ── ログキューをUIスレッドで消費 ──────────────────────────────
    def _drain_log_queue(self):
        try:
            while True:
                text, tag = self._log_queue.get_nowait()
                self._log_write(text, tag)
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _queue_log(self, text: str, tag: str = "info"):
        self._log_queue.put((text, tag))

    # ── ステータスポーリング ──────────────────────────────────────
    def _poll_status(self):
        fooocus_up = is_port_open(PORT_FOOOCUS)
        bridge_up  = is_port_open(PORT_BRIDGE)

        # Fooocus
        if fooocus_up:
            self._dot_fooocus.config(fg=GREEN)
            self._lbl_fooocus.config(text="起動中", fg=GREEN)
        else:
            if self._proc_fooocus and self._proc_fooocus.poll() is None:
                self._dot_fooocus.config(fg=YELLOW)
                self._lbl_fooocus.config(text="起動中...", fg=YELLOW)
            else:
                self._dot_fooocus.config(fg=RED)
                self._lbl_fooocus.config(text="停止中", fg=RED)

        # Bridge (Fooocusの中で起動されるためプロセスを別管理しない)
        if bridge_up:
            self._dot_bridge.config(fg=GREEN)
            self._lbl_bridge.config(text="起動中", fg=GREEN)
        else:
            self._dot_bridge.config(fg=RED)
            self._lbl_bridge.config(text="停止中", fg=RED)

        # ボタン状態
        running = (self._proc_fooocus is not None and
                   self._proc_fooocus.poll() is None)
        self._btn_start.config(state="disabled" if running else "normal")
        self._btn_stop.config(state="normal" if running else "disabled")
        self._btn_restart.config(state="normal" if running else "disabled")

        self.after(2000, self._poll_status)

    # ── 起動 ──────────────────────────────────────────────────────
    def _start_all(self):
        preset   = self._preset_var.get()
        browser  = self._browser_var.get()
        listen   = self._listen_var.get()

        cmd = build_fooocus_cmd(
            preset  = "" if preset == "なし" else preset,
            in_browser = browser,
            listen  = listen
        )

        self._log_write(f"▶ 起動コマンド: {' '.join(cmd)}", "accent")

        try:
            self._proc_fooocus = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )
        except FileNotFoundError as e:
            self._log_write(f"[ERROR] Python が見つかりません: {e}", "err")
            return

        self._log_write(f"PID: {self._proc_fooocus.pid}", "info")
        threading.Thread(target=self._stream_output,
                         args=(self._proc_fooocus,), daemon=True).start()

    def _stream_output(self, proc: subprocess.Popen):
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                low = line.lower()
                if any(k in low for k in ("error", "exception", "traceback", "failed")):
                    tag = "err"
                elif any(k in low for k in ("warning", "warn")):
                    tag = "warn"
                elif any(k in low for k in ("running on", "http://", "startup", "ready", "started")):
                    tag = "ok"
                else:
                    tag = "info"
                self._queue_log(line, tag)
        except Exception:
            pass
        self._queue_log("── プロセス終了 ──", "warn")

    # ── 停止 ──────────────────────────────────────────────────────
    def _stop_all(self):
        if self._proc_fooocus and self._proc_fooocus.poll() is None:
            self._log_write("■ 停止中...", "warn")
            try:
                self._proc_fooocus.terminate()
                try:
                    self._proc_fooocus.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self._proc_fooocus.kill()
                self._log_write("停止しました。", "ok")
            except Exception as e:
                self._log_write(f"[ERROR] 停止失敗: {e}", "err")
        self._proc_fooocus = None

    # ── 再起動 ────────────────────────────────────────────────────
    def _restart_all(self):
        self._log_write("↺ 再起動中...", "accent")
        self._stop_all()
        time.sleep(1)
        self._start_all()

    # ── 終了 ──────────────────────────────────────────────────────
    def _on_close(self):
        if (self._proc_fooocus and self._proc_fooocus.poll() is None):
            if messagebox.askyesno(
                "確認",
                "Fooocus が起動中です。終了しますか？\n(プロセスも停止されます)",
                parent=self
            ):
                self._stop_all()
            else:
                return
        self.destroy()


# ── エントリーポイント ─────────────────────────────────────────────
if __name__ == "__main__":
    app = LauncherApp()
    # ウィンドウを画面中央に
    app.update_idletasks()
    w, h = 700, 580
    sw = app.winfo_screenwidth()
    sh = app.winfo_screenheight()
    x  = (sw - w) // 2
    y  = (sh - h) // 2
    app.geometry(f"{w}x{h}+{x}+{y}")
    app.mainloop()
