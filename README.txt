
555-serverlite — Deploy rapido (Render)

1) File inclusi
   - 555-serverlite.patched.full.fixed.py  -> script completo per Render
   - requirements.txt                      -> dipendenze (pytz incluso)
   - README.txt                            

2) Variabili d'ambiente su Render (Settings → Environment)
   - TELEGRAM_BOT_TOKEN    = <token bot Telegram>
   - TELEGRAM_CHAT_ID      = @abkllr (o ID numerico/chat)
   - RENDER_EXTERNAL_URL   = https://five55a.onrender.com
   - GITHUB_TOKEN          = <token con scope gist>   (opzionale ma consigliato)
   - FLAGS_GIST_ID         = <ID del gist>            (non il nome file; nel gist ci sia daily_flags.json)

3) Build & Run
   - Inserisci questi file nel repo/service su Render.
   - Start command:  python 555-serverlite.patched.full.fixed.py
   - Porta: lo script espone un mini web server Flask; Render rileverà la porta automaticamente.

4) Note importanti
   - Scheduler:
     * 07:00  Rassegna stampa (6 pagine: 4 categorie, Analisi ML, Calendario+Outlook)
     * 08:10  Morning brief (headlines + EM FX & Commodities + mini spread sovrani)
     * 14:10  Lunch brief (stesso formato)
     * 20:10  Evening brief (stesso formato)
     * Recovery ogni 10' entro finestra (Rassegna 60', altri 80'). Debounce + cooldown anti-duplica.
   - Endpoint utili: / (ok), /health, /flags, /reset/flags (POST), /send/test
   - Se Gist non è configurato/valido, i flag sono salvati su file locale di Render.

5) Test veloce in locale (facoltativo)
   - pip install -r requirements.txt
   - python 555-serverlite.patched.full.fixed.py
   - Apri http://127.0.0.1:10000/health per verificare.

Buon deploy!
