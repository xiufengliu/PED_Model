# Heat Demand Analysis for Social Building in Denmark

Questo progetto contiene strumenti per l'analisi e la visualizzazione dei dati di consumo termico di un edificio sociale in Danimarca.

## Dataset

Il dataset `social_building_heat_demand_denmark.csv` contiene dati orari di consumo termico per un edificio sociale in Danimarca. I dati sono stati generati considerando:

- Variazioni stagionali (consumi più elevati in inverno, più bassi in estate)
- Profili giornalieri tipici (picchi mattutini e serali)
- Caratteristiche tipiche degli edifici residenziali danesi
- Effetti del clima danese sulle esigenze di riscaldamento

## Struttura del Dataset

Il dataset contiene le seguenti colonne:
- `timestamp`: Data e ora della misurazione (formato: YYYY-MM-DD HH:MM:SS)
- `heat_demand_kw`: Domanda di calore in kilowatt (kW)

## Caratteristiche del Consumo Termico

Il consumo termico dell'edificio sociale presenta le seguenti caratteristiche:

1. **Variazione Stagionale**: 
   - Inverno (dicembre-febbraio): Consumi elevati (180-250 kW)
   - Primavera (marzo-maggio): Consumi in diminuzione (60-200 kW)
   - Estate (giugno-agosto): Consumi minimi (20-55 kW)
   - Autunno (settembre-novembre): Consumi in aumento (60-210 kW)

2. **Profilo Giornaliero**:
   - Picco mattutino (6:00-9:00): Aumento del consumo quando gli occupanti si svegliano
   - Plateau diurno (9:00-16:00): Consumo relativamente stabile durante il giorno
   - Picco serale (17:00-21:00): Aumento del consumo quando gli occupanti tornano a casa
   - Valle notturna (22:00-5:00): Consumo ridotto durante la notte

## Script di Analisi

Lo script `heat_demand_analysis.py` genera le seguenti visualizzazioni:

1. **Consumo Medio Mensile**: Grafico a barre che mostra il consumo medio mensile
2. **Profilo Giornaliero per Stagione**: Grafico a linee che mostra il profilo giornaliero medio per ogni stagione
3. **Heatmap**: Mappa di calore che mostra il consumo per mese e ora del giorno

## Esecuzione dell'Analisi

Per eseguire l'analisi e generare le visualizzazioni:

```bash
python heat_demand_analysis.py
```

Le visualizzazioni verranno salvate nella directory `heat_demand_figures`.

## Utilizzo dei Dati nel Modello PED

Questi dati possono essere utilizzati nel modello PED (Positive Energy District) per:

1. Dimensionare correttamente i sistemi di riscaldamento
2. Ottimizzare la gestione energetica dell'edificio
3. Valutare l'impatto di misure di efficienza energetica
4. Stimare il potenziale di integrazione con fonti di energia rinnovabile

## Note Metodologiche

I dati sono stati generati basandosi su:
- Profili di carico termico tipici per edifici residenziali in Danimarca
- Dati climatici della Danimarca
- Caratteristiche costruttive tipiche degli edifici sociali danesi
- Standard di comfort termico europei
