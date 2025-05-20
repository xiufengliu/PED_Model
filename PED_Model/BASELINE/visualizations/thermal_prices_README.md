# Analisi dei Prezzi dell'Energia Termica in Danimarca

Questo progetto contiene strumenti per l'analisi e la visualizzazione dei dati di prezzo dell'energia termica in Danimarca, specificamente per il teleriscaldamento (district heating).

## Dataset

Il dataset `thermal_energy_prices_denmark.csv` contiene dati giornalieri dei prezzi dell'energia termica per il teleriscaldamento in Danimarca. I dati sono stati generati considerando:

- Variazioni stagionali (prezzi più elevati in inverno, più bassi in estate)
- Tendenze tipiche del mercato energetico danese
- Caratteristiche del sistema di teleriscaldamento danese
- Fattori economici e ambientali che influenzano i prezzi dell'energia termica

## Struttura del Dataset

Il dataset contiene le seguenti colonne:
- `Date`: Data della misurazione (formato: YYYY-MM-DD)
- `Price (EUR/MWh)`: Prezzo dell'energia termica in Euro per Megawattora

## Caratteristiche dei Prezzi dell'Energia Termica

I prezzi dell'energia termica in Danimarca presentano le seguenti caratteristiche:

1. **Variazione Stagionale**: 
   - Inverno (dicembre-febbraio): Prezzi elevati (75-114 EUR/MWh)
   - Primavera (marzo-maggio): Prezzi in diminuzione (63-93 EUR/MWh)
   - Estate (giugno-agosto): Prezzi minimi (32-63 EUR/MWh)
   - Autunno (settembre-novembre): Prezzi in aumento (33-93 EUR/MWh)

2. **Tendenza Annuale**:
   - I prezzi seguono un andamento ciclico con minimi in agosto e massimi in dicembre/gennaio
   - La variazione annuale riflette la domanda di riscaldamento, che è fortemente correlata alle temperature esterne

3. **Fattori di Influenza**:
   - Temperatura esterna (principale fattore)
   - Prezzo dei combustibili (gas naturale, biomassa)
   - Disponibilità di calore di scarto da impianti industriali e di incenerimento
   - Politiche energetiche e ambientali

## Script di Analisi

Lo script `thermal_prices_analysis.py` genera le seguenti visualizzazioni:

1. **Prezzi Medi Mensili**: Grafico a barre che mostra il prezzo medio mensile
2. **Tendenza dei Prezzi**: Grafico a linee che mostra l'andamento dei prezzi durante l'anno
3. **Confronto Stagionale**: Grafico a barre che confronta i prezzi medi, minimi e massimi per ogni stagione

## Esecuzione dell'Analisi

Per eseguire l'analisi e generare le visualizzazioni:

```bash
python thermal_prices_analysis.py
```

Le visualizzazioni verranno salvate nella directory `thermal_prices_figures`.

## Utilizzo dei Dati nel Modello PED

Questi dati possono essere utilizzati nel modello PED (Positive Energy District) per:

1. Valutare i costi operativi dei sistemi di riscaldamento
2. Ottimizzare le strategie di gestione energetica
3. Analizzare la fattibilità economica di diverse tecnologie di riscaldamento
4. Stimare i risparmi potenziali derivanti dall'implementazione di misure di efficienza energetica

## Note sul Sistema di Teleriscaldamento Danese

La Danimarca è leader mondiale nel teleriscaldamento, con oltre il 60% delle abitazioni danesi collegate a reti di teleriscaldamento. Il sistema danese è caratterizzato da:

1. **Alta Efficienza**: Utilizzo di cogenerazione (CHP) e recupero del calore di scarto
2. **Fonti Rinnovabili**: Crescente integrazione di biomassa, energia solare termica e pompe di calore
3. **Regolamentazione**: Prezzi regolati basati sui costi effettivi (no-profit)
4. **Proprietà Comunale**: La maggior parte delle reti è di proprietà comunale o cooperativa
5. **Innovazione**: Implementazione di reti a bassa temperatura e integrazione con altre reti energetiche

Questi fattori contribuiscono a rendere il sistema di teleriscaldamento danese uno dei più efficienti e sostenibili al mondo, con prezzi relativamente stabili e competitivi rispetto ad altre opzioni di riscaldamento.
