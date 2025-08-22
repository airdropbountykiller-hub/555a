# 🚀 GitHub Actions Wake-Up System per 555 Render

## 📂 File Creati

- `wakeup.yml` → Wake-up automatico programmato
- `emergency-reset.yml` → Azioni di emergenza manuali

## 🔧 Setup su GitHub

### 1. **Copia nel Repository**
Copia questi file nella struttura:
```
your-repo/
├── .github/
│   └── workflows/
│       ├── wakeup.yml
│       └── emergency-reset.yml
```

### 2. **Push al Repository**
```bash
git add .github/workflows/
git commit -m "Add GitHub Actions wake-up system"
git push
```

### 3. **Verifica Attivazione**
- Vai su GitHub → Repository → **Actions** tab
- Dovresti vedere i workflow attivi

## ⏰ **Schedule Automatico**

### **wakeup.yml** - Wake-up Programmati
- **06:50 CET** → Pre-rassegna stampa (07:00)
- **08:05 CET** → Pre-morning report (08:10)  
- **14:05 CET** → Pre-lunch report (14:10)
- **20:05 CET** → Pre-evening report (20:10)

### **emergency-reset.yml** - Solo Manuale
- **Wake + Reset** → Sveglia app e resetta flag
- **Wake Only** → Solo sveglia app
- **Reset Only** → Solo resetta flag

## 🌍 **Orari UTC vs CET**

Il sistema gestisce **automaticamente** estate/inverno:
- **Estate (CET = UTC+2)**: `cron: "50 4 * * *"` = 06:50 CET
- **Inverno (CET = UTC+1)**: `cron: "50 5 * * *"` = 06:50 CET

GitHub esegue **entrambi** i cron, ma uno sarà sempre fuori orario.

## 🔥 **Come Usare Emergency**

1. **Vai su GitHub** → Repository → **Actions**
2. **Clicca** "555 Emergency Reset & Wake-Up"
3. **Run workflow** → Scegli azione:
   - `wake_and_reset` → Sveglia + resetta flag (raccomandato)
   - `wake_only` → Solo sveglia (se app dorme)
   - `reset_only` → Solo reset flag (se bloccati)

## ✅ **Vantaggi**

- **Gratuito** → 2000 min/mese GitHub Actions
- **Affidabile** → Infrastruttura enterprise GitHub
- **IP puliti** → Non bloccati da Render
- **Controllo totale** → Manuale + automatico
- **Monitoring** → Log completi di ogni azione

## 🎯 **Risultato Atteso**

Con questo sistema, il tuo 555-lite su Render dovrebbe:
- ✅ **Svegliarsi sempre** prima degli orari programmati
- ✅ **Inviare puntualmente** tutti i report
- ✅ **Recovery di emergenza** disponibile 24/7

## 🚨 **Note Importanti**

- I **cron** GitHub Actions hanno precisione di ±15 minuti nei picchi di traffico
- **Workflow_dispatch** permette attivazione istantanea manuale
- **Timeout** impostati per evitare job infiniti
- **User-Agent** vari per sembrare traffico normale
