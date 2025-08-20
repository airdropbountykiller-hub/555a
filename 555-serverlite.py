#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
555-serverlite.patched.full.py
Lite per Render con:
- /health, /files/list, /files/raw/<name>
- flags giornalieri su Gist (daily_flags.json) con fallback locale
- orari: 07:00 rassegna, 08:10 morning, 14:10 lunch, 20:10 evening
- recovery ogni 10 minuti con debounce
"""
import os, json, time, threading, datetime
from typing import Dict, Any
import pytz, requests
from flask import Flask, jsonify, send_file, request, abort

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
GITHUB_TOKEN       = os.getenv("GITHUB_TOKEN", "").strip()
FLAGS_GIST_ID_RAW  = os.getenv("FLAGS_GIST_ID", "").strip()
RENDER_EXTERNAL_URL= os.getenv("RENDER_EXTERNAL_URL", "").strip()
PULL_SECRET        = os.getenv("PULL_SECRET", "").strip()
PORT               = int(os.getenv("PORT", "10000"))

ITALY_TZ = pytz.timezone("Europe/Rome")
SAVE_DIR = "salvataggi"
os.makedirs(SAVE_DIR, exist_ok=True)

def _extract_gist_id(inp: str) -> str:
    if not inp: return ""
    if "/" not in inp and " " not in inp:
        return inp.strip()
    parts = inp.split("/")
    return parts[-1].split("?")[0].split("#")[0].strip()

FLAGS_GIST_ID = _extract_gist_id(FLAGS_GIST_ID_RAW)

GLOBAL_FLAGS: Dict[str, Any] = {
    "rassegna_sent": False,
    "morning_news_sent": False,
    "daily_report_sent": False,
    "evening_report_sent": False,
    "last_reset_date": datetime.datetime.now(ITALY_TZ).strftime("%Y%m%d"),
    "rassegna_last_run": "",
    "morning_last_run": "",
    "lunch_last_run": "",
    "evening_last_run": "",
}

def _today_key(dt): return dt.strftime("%Y%m%d")
def _minute_key(dt): return dt.strftime("%Y%m%d%H%M")
def _now_it(): return datetime.datetime.now(ITALY_TZ)

def _flags_path_local(): return os.path.join(SAVE_DIR, "daily_flags.json")
def _load_flags_local():
    try:
        with open(_flags_path_local(), "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}
def _save_flags_local(d):
    with open(_flags_path_local(), "w", encoding="utf-8") as f: json.dump(d, f, ensure_ascii=False, indent=2)

def _gist_headers():
    return {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
def _gist_get():
    if not FLAGS_GIST_ID: raise RuntimeError("FLAGS_GIST_ID mancante")
    r = requests.get(f"https://api.github.com/gists/{FLAGS_GIST_ID}", headers=_gist_headers(), timeout=15)
    if r.status_code == 404: raise FileNotFoundError("Gist non trovato (404)")
    r.raise_for_status(); return r.json()
def _ensure_gist_has_daily_flags():
    try:
        data = _gist_get()
        files = data.get("files", {})
        entry = files.get("daily_flags.json")
        content = (entry or {}).get("content", "")
        if not content.strip():
            payload = {"files": {"daily_flags.json": {"content": json.dumps({}, ensure_ascii=False, indent=2)}}}
            p = requests.patch(f"https://api.github.com/gists/{FLAGS_GIST_ID}", headers=_gist_headers(), json=payload, timeout=15)
            p.raise_for_status()
            return _gist_get()
        try:
            json.loads(content); return data
        except Exception:
            payload = {"files": {"daily_flags.json": {"content": json.dumps({}, ensure_ascii=False, indent=2)}}}
            p = requests.patch(f"https://api.github.com/gists/{FLAGS_GIST_ID}", headers=_gist_headers(), json=payload, timeout=15)
            p.raise_for_status()
            return _gist_get()
    except Exception:
        return {}

def _load_daily_flags():
    try:
        if FLAGS_GIST_ID and GITHUB_TOKEN:
            data = _ensure_gist_has_daily_flags()
            files = data.get("files", {}) if data else {}
            entry = files.get("daily_flags.json")
            if entry and entry.get("content", "").strip():
                return json.loads(entry["content"])
    except Exception as e:
        print(f"⚠️ [FLAGS-GIST] load: {e}")
    return _load_flags_local() or {}

def _save_daily_flags(d: Dict[str, Any]):
    _save_flags_local(d)
    try:
        if FLAGS_GIST_ID and GITHUB_TOKEN:
            payload = {"files": {"daily_flags.json": {"content": json.dumps(d, ensure_ascii=False, indent=2)}}}
            p = requests.patch(f"https://api.github.com/gists/{FLAGS_GIST_ID}", headers=_gist_headers(), json=payload, timeout=15)
            p.raise_for_status()
    except Exception as e:
        print(f"⚠️ [FLAGS-GIST] save: {e}")

def _daily_key(base, now): return f"{base}_sent_{_today_key(now)}"
def _is_sent_today(base, now): return bool(_load_daily_flags().get(_daily_key(base, now), False))
def _mark_sent_today(base, now):
    d = _load_daily_flags(); d[_daily_key(base, now)] = True; _save_daily_flags(d)

def _send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ [Telegram] TOKEN/CHAT_ID mancanti"); return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 400:
            r = requests.post(url, json={**payload, "parse_mode": None}, timeout=20)
        r.raise_for_status(); return True
    except Exception as e:
        print(f"❌ [Telegram] invio: {e}"); return False

def generate_rassegna_stampa():
    return ("🗞️ *Rassegna Stampa 6p*\n"
            "1) Macro\n2) Mercati\n3) Tech\n4) Geo\n"
            "5) 🔎 ML Notizie: 5 rischi\n"
            "6) 📅 Calendario + ML: 5 eventi")

def generate_morning_news():
    return ("🌅 *Morning Brief*\n• 2-3 headline\n• EM FX & Commodities\n• Spread sovrani")

def generate_lunch_report():
    return ("🍝 *Lunch*\n• Indicatori chiave\n• EM FX & Commodities\n• 2-3 headline\n• Spread sovrani")

def generate_evening_report():
    return ("🌙 *Evening*\n• Resoconto\n• EM FX & Commodities\n• 2-3 headline\n• Spread sovrani")

def _reset_if_new_day(now):
    if GLOBAL_FLAGS["last_reset_date"] != _today_key(now):
        GLOBAL_FLAGS["last_reset_date"] = _today_key(now)
        for k in ["rassegna_sent","morning_news_sent","daily_report_sent","evening_report_sent"]:
            GLOBAL_FLAGS[k] = False
        _save_daily_flags({})

def _try_send(base, fn, now, debounce_key):
    minute_key = _minute_key(now)
    if GLOBAL_FLAGS.get(debounce_key) == minute_key: return
    GLOBAL_FLAGS[debounce_key] = minute_key
    if _is_sent_today(base, now): return
    try:
        msg = fn()
        if _send_telegram(msg):
            _mark_sent_today(base, now)
            print(f"✅ [{base}] inviato {now.strftime('%H:%M')}")
    except Exception as e:
        print(f"❌ [{base}] errore: {e}")

def scheduler_loop():
    print("🚀 [LITE] Scheduler ON")
    while True:
        now = _now_it(); hhmm = now.strftime("%H:%M"); _reset_if_new_day(now)

        if hhmm == "07:00":
            _try_send("rassegna", generate_rassegna_stampa, now, "rassegna_last_run"); time.sleep(60)
        elif hhmm == "08:10":
            _try_send("morning_news", generate_morning_news, now, "morning_last_run"); time.sleep(60)
        elif hhmm == "14:10":
            _try_send("daily_report", generate_lunch_report, now, "lunch_last_run"); time.sleep(60)
        elif hhmm == "20:10":
            _try_send("evening_report", generate_evening_report, now, "evening_last_run"); time.sleep(60)

        if now.minute % 10 == 0 and now.second < 3:
            if hhmm > "07:00" and not _is_sent_today("rassegna", now):
                _try_send("rassegna", generate_rassegna_stampa, now, "rassegna_last_run")
            if hhmm > "08:10" and not _is_sent_today("morning_news", now):
                _try_send("morning_news", generate_morning_news, now, "morning_last_run")
            if hhmm > "14:10" and not _is_sent_today("daily_report", now):
                _try_send("daily_report", generate_lunch_report, now, "lunch_last_run")
            if hhmm > "20:10" and not _is_sent_today("evening_report", now):
                _try_send("evening_report", generate_evening_report, now, "evening_last_run")
            time.sleep(3)

        time.sleep(0.5)

app = Flask(__name__)

def _auth_ok():
    return True if not PULL_SECRET else request.headers.get("X-Pull-Secret","") == PULL_SECRET

@app.route("/")
def root(): return "555-lite OK"

@app.route("/health")
def health(): return "OK", 200

@app.route("/files/list")
def files_list():
    if not _auth_ok(): abort(401)
    out = []
    for nm in sorted(os.listdir(SAVE_DIR)):
        p = os.path.join(SAVE_DIR, nm)
        if os.path.isfile(p): out.append({"name": nm, "size": os.path.getsize(p)})
    return jsonify({"files": out, "count": len(out)})

@app.route("/files/raw/<path:name>")
def files_raw(name):
    if not _auth_ok(): abort(401)
    p = os.path.join(SAVE_DIR, name)
    if not os.path.isfile(p): abort(404)
    return send_file(p, as_attachment=True)

def run_server():
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

def main():
    print("🌐 [LITE] Web server su porta", PORT)
    try:
        flags = _load_daily_flags()
        if isinstance(flags, dict):
            today = _today_key(_now_it())
            for k, v in flags.items():
                if k.endswith(today) and v is True:
                    if k.startswith("rassegna_sent"): GLOBAL_FLAGS["rassegna_sent"] = True
                    if k.startswith("morning_news_sent"): GLOBAL_FLAGS["morning_news_sent"] = True
                    if k.startswith("daily_report_sent"): GLOBAL_FLAGS["daily_report_sent"] = True
                    if k.startswith("evening_report_sent"): GLOBAL_FLAGS["evening_report_sent"] = True
    except Exception as e:
        print("⚠️ [INIT-FLAGS]", e)

    th = threading.Thread(target=scheduler_loop, daemon=True); th.start()
    run_server()

if __name__ == "__main__":
    main()
