import pandas as pd

# --- 1. Leggi il file Excel ---
excel_file = "samlet nye data 2023 samle sheets only.xlsb"
sheet_name = "Foglio1"
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# --- 2. Controlla presenza colonne attese ---
if 'timestamp' not in df.columns:
    raise ValueError("La colonna 'timestamp' non è presente.")

# Trova il nome della colonna dei prezzi se non è 'price'
price_col = [col for col in df.columns if 'price' in col.lower()]
if not price_col:
    raise ValueError("Colonna del prezzo non trovata.")
df = df.rename(columns={price_col[0]: 'price'})

# --- 3. Parsing timestamp e arrotondamento ---
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.floor('h')
df['timestamp'] = df['timestamp'].apply(lambda x: x.replace(year=2025) if pd.notnull(x) else x)

# --- 4. Normalizza i prezzi (virgola a punto) ---
df['price'] = df['price'].astype(str).str.replace(',', '.')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# --- 5. Rimuovi righe invalide o duplicate ---
df = df.dropna(subset=['timestamp', 'price'])
df = df.drop_duplicates(subset='timestamp')

# --- 6. Crea tutte le ore del 2025 ---
full_index = pd.date_range('2025-01-01 00:00:00', '2025-12-31 23:00:00', freq='h')
df_full = pd.DataFrame(index=full_index)
df_full.index.name = 'timestamp'

# --- 7. Inserisce i prezzi veri dove presenti ---
df = df.set_index('timestamp')
df_final = df_full.join(df, how='left')

# --- 8. Esporta il risultato ---
df_final.to_csv("dati_2025_completi.csv")

print("✅ File creato con successo con", df_final['price'].count(), "prezzi associati.")
