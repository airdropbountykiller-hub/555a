# -*- coding: utf-8 -*-
"""
555-serverlite.deploy.py — Single-file for Render
Features:
- Flask app with /health and secure read-only /files/* endpoints (X-Pull-Secret optional)
- Daily flags with optional Gist (FLAGS_GIST_ID + GITHUB_TOKEN) and local fallback
- Schedules (Europe/Rome):
    07:00  RASSEGNA_STAMPA   (sets flag: 'rassegna_stampa')
    08:10  MORNING            (flag: 'morning_news')
    14:10  LUNCH              (flag: 'daily_report')
    20:10  EVENING            (flag: 'evening_report')
- Recovery: every 10 minutes checks missed jobs already past their scheduled time (until 23:59)
- Robust Telegram sender (Markdown with fallback to plain)
- All content generators are optional: if your original functions exist (e.g. generate_rassegna_stampa_sixpages)
  they are called; otherwise a minimal fallback message is sent.
"""

import os, json, time, datetime, threading, pytz, requests, base64
from pathlib import Path
from flask import Flask, jsonify, request, abort, send_file

ITALY_TZ = pytz.timezone("Europe/Rome")
BASE_DIR = Path(__file__).resolve().parent
SALVATAGGI = (BASE_DIR / "salvataggi"); SALVATAGGI.mkdir(parents=True, exist_ok=True)
FLAGS_PATH = SALVATAGGI / "daily_flags.json"

# ----------------- ENV -----------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN","").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID","").strip()
RENDER_EXTERNAL_URL= os.getenv("RENDER_EXTERNAL_URL","").strip()
GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN","").strip()
FLAGS_GIST_ID      = os.getenv("FLAGS_GIST_ID","").strip()
PULL_SECRET        = os.getenv("PULL_SECRET","").strip()

# ----------------- Helpers -----------------
def _now():
    return datetime.datetime.now(ITALY_TZ)

def _today_key(dt=None):
    dt = dt or _now()
    return dt.strftime("%Y%m%d")

def _ts(dt):
    return dt.strftime("%H:%M:%S")

def _escape_md_v1(text: str) -> str:
    repl = {
        '_': r'\_', '*': r'\*', '[': r'\[', ']': r'\]', '(': r'\(', ')': r'\)',
        '~': r'\~', '`': r'\`', '>': r'\>', '#': r'\#', '+': r'\+', '-': r'\-',
        '=': r'\=', '|': r'\|', '{': r'\{', '}': r'\}', '.': r'\.', '!': r'\!',
    }
    return ''.join(repl.get(ch, ch) for ch in text)

def log(msg):
    print(msg, flush=True)

# ----------------- Flags (local + optional Gist) -----------------
DEFAULT_FLAGS = {
    "date": _today_key(),
    "rassegna_stampa_sent": False,
    "rassegna_stampa_last_run": "",
    "morning_news_sent": False,
    "morning_news_last_run": "",
    "daily_report_sent": False,
    "daily_report_last_run": "",
    "evening_report_sent": False,
    "evening_report_last_run": ""
}

def _load_flags_local():
    if not FLAGS_PATH.exists():
        FLAGS_PATH.write_text(json.dumps(DEFAULT_FLAGS, indent=2, ensure_ascii=False), encoding="utf-8")
        log("📁 [FLAGS-FILE] Creato daily_flags.json locale")
    try:
        data = json.loads(FLAGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = DEFAULT_FLAGS.copy()
    if data.get("date") != _today_key():
        data = DEFAULT_FLAGS.copy(); data["date"] = _today_key()
    return data

def _save_flags_local(data):
    FLAGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log("💾 [FLAGS-FILE] Flag salvati su file locale")

def _load_flags_gist():
    if not (GITHUB_TOKEN and FLAGS_GIST_ID):
        raise RuntimeError("Gist non configurato")
    import requests
    r = requests.get(f"https://api.github.com/gists/{FLAGS_GIST_ID}", headers={"Authorization": f"token {GITHUB_TOKEN}"} , timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"Errore gist GET: {r.status_code}")
    j = r.json(); files = j.get("files",{})
    if "daily_flags.json" not in files:
        raise RuntimeError("daily_flags.json non presente nel gist")
    content = files["daily_flags.json"].get("content","{}")
    data = json.loads(content)
    if data.get("date") != _today_key():
        data = DEFAULT_FLAGS.copy(); data["date"] = _today_key()
    return data

def _save_flags_gist(data):
    if not (GITHUB_TOKEN and FLAGS_GIST_ID):
        raise RuntimeError("Gist non configurato")
    import requests
    payload = {"files": {"daily_flags.json": {"content": json.dumps(data, indent=2, ensure_ascii=False)}}}
    r = requests.patch(f"https://api.github.com/gists/{FLAGS_GIST_ID}", headers={"Authorization": f"token {GITHUB_TOKEN}"}, json=payload, timeout=15)
    if r.status_code not in (200,201):
        raise RuntimeError(f"Errore gist PATCH: {r.status_code} {r.text[:200]}")
    return True

def load_flags():
    try:
        data = _load_flags_gist()
        log("✅ [FLAGS-GIST] Caricati dal gist")
        _save_flags_local(data)
        return data
    except Exception as e:
        log(f"⚠️ [FLAGS-GIST] {e}")
        data = _load_flags_local()
        return data

def save_flags(data):
    _save_flags_local(data)
    try:
        _save_flags_gist(data)
        log("✅ [FLAGS-GIST] Aggiornati su gist")
    except Exception as e:
        log(f"⚠️ [FLAGS-GIST] Salvataggio gist fallito: {e}")

def is_sent(flag_key):
    d = load_flags()
    return bool(d.get(f"{flag_key}_sent", False))

def mark_sent(flag_key):
    d = load_flags()
    d[f"{flag_key}_sent"] = True
    d[f"{flag_key}_last_run"] = _now().strftime("%Y%m%d%H%M")
    save_flags(d)

def debounce_minute(flag_key):
    d = load_flags()
    now_key = _now().strftime("%Y%m%d%H%M")
    last = d.get(f"{flag_key}_last_run","")
    if last == now_key:
        return False
    d[f"{flag_key}_last_run"] = now_key
    save_flags(d)
    return True

# ----------------- Telegram -----------------
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("🛑 [Telegram] Token/chat non configurati"); return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": _escape_md_v1(text), "parse_mode": "Markdown", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            log("✅ [Telegram] Inviato (Markdown)"); return True
        log(f"⚠️ [Telegram] {r.status_code} {r.text[:160]} — retry plain")
        payload.pop("parse_mode", None)
        r2 = requests.post(url, json=payload, timeout=15)
        if r2.status_code == 200:
            log("✅ [Telegram] Inviato (plain)"); return True
        log(f"❌ [Telegram] Fallback {r2.status_code} {r2.text[:160]}")
    except Exception as e:
        log(f"❌ [Telegram] Errore invio: {e}")
    return False

# ----------------- Generatori (hook) -----------------
def _call_generator(name: str, fallback_title: str):
    try:
        fn = globals().get(name)
        if callable(fn):
            return fn()
    except Exception as e:
        log(f"❌ [GEN] Errore in {name}: {e}")
    # Fallback minimale
    return f"{fallback_title}\n\n(attiva la funzione {name} nel codice per il contenuto completo)"

def gen_rassegna():
    return _call_generator("generate_rassegna_stampa_sixpages", "🗞️ RASSEGNA STAMPA (fallback)")

def gen_morning():
    return _call_generator("generate_morning_news_briefing", "🌅 MORNING BRIEF (fallback)")

def gen_lunch():
    return _call_generator("generate_lunch_report", "🍽️ LUNCH REPORT (fallback)")

def gen_evening():
    return _call_generator("generate_evening_report", "🌙 EVENING REPORT (fallback)")

# ----------------- Schedules -----------------
SCHEDULES = {
    "rassegna_stampa": {"time":"07:00", "gen":gen_rassegna, "title":"rassegna_stampa"},
    "morning_news":    {"time":"08:10", "gen":gen_morning, "title":"morning_news"},
    "daily_report":    {"time":"14:10", "gen":gen_lunch,   "title":"daily_report"},
    "evening_report":  {"time":"20:10", "gen":gen_evening, "title":"evening_report"},
}
RECOVERY_MINUTES = {0,10,20,30,40,50}  # ogni 10 minuti

def _time_eq(hhmm: str) -> bool:
    return _now().strftime("%H:%M") == hhmm

def _after(hhmm: str) -> bool:
    now = _now()
    target = now.replace(hour=int(hhmm[:2]), minute=int(hhmm[3:]), second=0, microsecond=0)
    return now >= target

def run_job(key: str):
    cfg = SCHEDULES[key]
    title = cfg["title"]
    if is_sent(title):
        return
    if not debounce_minute(title):  # anti-rientro nello stesso minuto
        return
    log(f"🚀 [JOB] Avvio {key} alle {_ts(_now())}")
    try:
        text = cfg["gen"]()
        ok = send_telegram(text)
        if ok:
            mark_sent(title); log(f"✅ [JOB] {key} inviato")
        else:
            log(f"❌ [JOB] {key} invio fallito")
    except Exception as e:
        log(f"❌ [JOB] {key} errore: {e}")

def scheduler_loop():
    log("🚀 [LITE-MAIN] Scheduler attivo")
    while True:
        try:
            now = _now()
            hhmm = now.strftime("%H:%M")
            mm = now.minute
            # reset flags a inizio nuovo giorno
            d = load_flags()
            if d.get("date") != _today_key():
                d.update(DEFAULT_FLAGS); d["date"] = _today_key(); save_flags(d)
            # orari principali
            for key, cfg in SCHEDULES.items():
                if _time_eq(cfg["time"]):
                    run_job(key)
            # recovery ogni 10 min dopo l'orario
            if mm in RECOVERY_MINUTES:
                for key, cfg in SCHEDULES.items():
                    if (not is_sent(cfg["title"])) and _after(cfg["time"]):
                        log(f"🔄 [RECOVERY] Tentativo per {key}")
                        run_job(key)
            time.sleep(5)
        except Exception as e:
            log(f"❌ [SCHED] errore loop: {e}")
            time.sleep(2)

# ----------------- Flask App + Pull API -----------------
app = Flask(__name__)

@app.route("/")
def root():
    return "555-lite OK"

@app.route("/health")
def health():
    return "OK", 200

# Secure read-only file endpoints
ALLOWED = {
    "segnali_tecnici.csv",
    "previsioni_ml.csv",
    "previsioni_cumulativo.csv",
    "indicatori_cumulativo.csv",
    "calendario_eventi.csv",
    "daily_flags.json",
    "rassegna_stampa_manuale.csv",
    "rassegna_stampa_manuale.txt",
    "analysis_text.txt",
}
def _auth():
    if not PULL_SECRET:
        return True
    got = request.headers.get("X-Pull-Secret","").strip()
    if got != PULL_SECRET:
        abort(401)
    return True

@app.get("/files/ping")
def files_ping():
    _auth()
    return jsonify({"ok":True, "salvataggi": str(SALVATAGGI)})

@app.get("/files/list")
def files_list():
    _auth()
    items = []
    for name in sorted(ALLOWED):
        p = (SALVATAGGI / name)
        if p.exists():
            st = p.stat()
            items.append({"name": name, "size": st.st_size, "mtime": int(st.st_mtime)})
    return jsonify({"base": str(SALVATAGGI), "count": len(items), "files": items})

@app.get("/files/raw/<path:name>")
def files_raw(name):
    _auth()
    p = (SALVATAGGI / name).resolve()
    if SALVATAGGI not in p.parents and p != SALVATAGGI: abort(400)
    if p.name not in ALLOWED: abort(404)
    if not p.exists(): abort(404)
    return send_file(str(p), as_attachment=False, download_name=p.name)

def _start_threads():
    t = threading.Thread(target=scheduler_loop, daemon=True); t.start()
    log("🚀 [THREADS] Scheduler avviato")

if __name__ == "__main__":
    _start_threads()
    port = int(os.getenv("PORT","8000"))
    log(f"🌐 [555-LITE] Web server su porta {port}")
    app.run(host="0.0.0.0", port=port)
