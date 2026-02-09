import json
import requests
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from astroquery.gaia import Gaia
import math
import sys

# CONFIGURAZIONE
OUTPUT_FILE = "universe_db.json"
MAX_OBJECTS = 2000 # Limite per non far esplodere GitHub (puoi aumentarlo)

print("--- PHOENIX V100: AUTONOMOUS ANOMALY HUNTER ---")

# ==========================================
# 1. MODULO DI INGESTIONE DATI (HARVESTER)
# ==========================================
def get_nasa_data():
    print("[INGESTION] Connessione NASA Archive...")
    # Scarichiamo: Massa, Raggio, Temperatura, Distanza, Densità, Anno Scoperta
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+top+1000+pl_name,pl_bmasse,pl_rade,pl_eqt,sy_dist,pl_dens,ra,dec,disc_year+from+ps+where+sy_dist>0+and+pl_bmasse>0+order+by+disc_year+desc&format=json"
    try:
        data = requests.get(url).json()
        clean_data = []
        for p in data:
            # Pulizia Dati (Rimuoviamo i valori nulli per l'AI)
            if p['pl_bmasse'] and p['pl_rade'] and p['sy_dist']:
                clean_data.append({
                    "name": p['pl_name'],
                    "mass": float(p['pl_bmasse']),
                    "rad": float(p['pl_rade']),
                    "temp": float(p['pl_eqt']) if p['pl_eqt'] else 0,
                    "dist": float(p['sy_dist']),
                    "ra": float(p['ra']),
                    "dec": float(p['dec']),
                    "src": "NASA"
                })
        print(f" -> Scaricati {len(clean_data)} esopianeti.")
        return clean_data
    except Exception as e:
        print(f"[ERROR] NASA Down: {e}")
        return []

def get_gaia_data():
    print("[INGESTION] Connessione ESA Gaia DR3...")
    # Scarichiamo stelle con parametri fisici completi
    query = """
    SELECT TOP 1000 source_id, ra, dec, parallax, phot_g_mean_mag, teff_gspphot, logg_gspphot
    FROM gaiadr3.gaia_source 
    WHERE parallax > 5 AND teff_gspphot IS NOT NULL
    """
    try:
        job = Gaia.launch_job(query)
        res = job.get_results()
        clean_data = []
        for row in res:
            dist = 1000.0 / float(row['parallax'])
            clean_data.append({
                "name": f"GAIA-{str(row['source_id'])[:8]}",
                "mass": float(row['logg_gspphot']) * 10 if row['logg_gspphot'] else 1.0, # LogG come proxy massa
                "rad": float(row['phot_g_mean_mag']), # Mag come proxy raggio (inverso)
                "temp": float(row['teff_gspphot']),
                "dist": dist,
                "ra": float(row['ra']),
                "dec": float(row['dec']),
                "src": "GAIA"
            })
        print(f" -> Scaricate {len(clean_data)} sorgenti stellari.")
        return clean_data
    except Exception as e:
        print(f"[ERROR] GAIA Down: {e}")
        return []

# ==========================================
# 2. IL CERVELLO (AI NON SUPERVISIONATA)
# ==========================================
def analyze_universe(objects):
    print("[AI CORE] Inizializzazione Analisi Statistica...")
    
    if len(objects) < 10:
        print("Dati insufficienti per l'AI.")
        return objects

    # Preparazione Matrice Numerica per l'AI [Massa, Raggio, Temp, Distanza]
    # L'AI legge solo i numeri, non i nomi.
    features = []
    for o in objects:
        features.append([o['mass'], o['rad'], o['temp'], o['dist']])
    
    X = np.array(features)
    
    # 2A. NORMALIZZAZIONE (StandardScaler)
    # L'AI deve capire che 5000 gradi è "tanto" e 0.1 raggio è "poco".
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2B. ANOMALY DETECTION (Isolation Forest)
    # Cerca oggetti che si "isolano" matematicamente dal resto
    print("[AI CORE] Caccia alle Anomalie (Isolation Forest)...")
    iso = IsolationForest(contamination=0.05, random_state=42) # Top 5% più strani
    anomaly_labels = iso.fit_predict(X_scaled) # -1 = Anomalia, 1 = Normale
    anomaly_scores = iso.decision_function(X_scaled) # Più basso è, più è anomalo
    
    # 2C. CLUSTERING AUTOMATICO (K-Means)
    # Raggruppa l'universo in 5 famiglie logiche (senza sapere cosa sono)
    print("[AI CORE] Auto-Classificazione (K-Means Clustering)...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 2D. INTEGRAZIONE RISULTATI
    enriched_objects = []
    scale_3d = 100 # Fattore scala per Three.js

    for i, obj in enumerate(objects):
        # Calcolo Coordinate 3D
        d = obj['dist'] * scale_3d
        ra_rad = math.radians(obj['ra'])
        dec_rad = math.radians(obj['dec'])
        
        x = d * math.cos(dec_rad) * math.cos(ra_rad)
        y = d * math.sin(dec_rad)
        z = d * math.cos(dec_rad) * math.sin(ra_rad)
        
        # Logica AI
        is_anomaly = anomaly_labels[i] == -1
        score = abs(anomaly_scores[i]) # Forza del segnale anomalo
        cluster_id = int(clusters[i])
        
        # Assegnazione Colori AI
        if is_anomaly:
            color = "#ff0000" # ROSSO PURO = ANOMALIA GRAVE
            tag = f"ANOMALY DETECTED (Score {score:.2f})"
            radius_viz = 2.0 # Più grande nel visore
        else:
            # Colori basati sul cluster (Famiglie)
            # L'AI decide i colori per gruppo
            colors = ["#00ffaa", "#0088ff", "#ffff00", "#ff00ff", "#ffffff"] 
            color = colors[cluster_id % len(colors)]
            tag = f"CLASS-{cluster_id} OBJECT"
            radius_viz = 1.0

        # Costruzione Oggetto Finale JSON
        enriched_objects.append({
            "n": obj['name'],
            "cat": "ANOMALY" if is_anomaly else f"CLUSTER {cluster_id}",
            "t": tag,
            "col": color,
            "x": int(x), "y": int(y), "z": int(z),
            "r": radius_viz * (obj['rad'] if obj['rad'] < 10 else 10), # Cap raggio visivo
            "d": {
                "Massa": f"{obj['mass']:.2f}",
                "Temp": f"{obj['temp']:.0f} K",
                "Anomaly Score": f"{score:.4f}",
                "AI Cluster": str(cluster_id),
                "Source": obj['src']
            }
        })
        
    # Ordiniamo per mettere le anomalie alla fine (così si renderizzano sopra)
    enriched_objects.sort(key=lambda k: k['t'].startswith("ANOMALY"))
    return enriched_objects

# ==========================================
# 3. ESECUZIONE
# ==========================================
def main():
    # 1. Scarica tutto
    data_nasa = get_nasa_data()
    data_gaia = get_gaia_data()
    full_raw = data_nasa + data_gaia
    
    # 2. L'AI processa tutto insieme
    final_db = analyze_universe(full_raw)
    
    # 3. Salva
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_db, f)
    
    print(f"[COMPLETE] Analizzati {len(full_raw)} oggetti.")
    print(f"[COMPLETE] Database '{OUTPUT_FILE}' generato.")

if __name__ == "__main__":
    main()