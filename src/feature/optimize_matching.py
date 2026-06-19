import json
import pulp

# Carica i dati di matching dal file json
with open("data/raw/matching_dict_raw.json", "r") as f:
    matching_data = json.load(f)

all_wmd_values = [wmd for candidate_dicts in matching_data.values() for wmd in candidate_dicts.values()]
if all_wmd_values:
    max_wmd = max(all_wmd_values)
    matching_wmd_value = max_wmd + 1
else:
    matching_wmd_value = 10

print(f"matching_wmd_value == {matching_wmd_value}\n")

# Inizializza il problema di ottimizzazione lineare (Massimizzazione)
prob = pulp.LpProblem("Ottimizzazione_Matching_Reddit", pulp.LpMaximize)
# (splver cercherà valore più alto possibile con la formula che gli diamo) (greedy af)

# Crea le variabili decisionali binarie x_ij per le coppie valide
x = {}
for anchor, candidates in matching_data.items():
    x[anchor] = {}
    for candidate in candidates:
        x[anchor][candidate] = pulp.LpVariable(
            f"x_{anchor}_{candidate}", cat="Binary" # o 0 o 1 (rifiutato o accettato NO ALTERNATIVEEEEEEE)
        )

# Definisce la funzione obiettivo combinata
# Usiamo la costante per dare priorità assoluta al numero di match stabili
obj_terms = []
for anchor, candidates in matching_data.items():
    for candidate, wmd in candidates.items():
        obj_terms.append((matching_wmd_value - wmd) * x[anchor][candidate])
        # così se xij è 0 (interruttore spento) match non viene attivato
prob += pulp.lpSum(obj_terms)

# Vincolo 1: Ogni anchor può essere associato al massimo a un candidato (somma per riga <= 1)
for anchor in matching_data:
    prob += (
        pulp.lpSum(x[anchor][candidate] for candidate in matching_data[anchor])
        <= 1
    )
# perchè gle sto a dì che dato che la somma di tutti gli interruttori dev'essere <=1
# deve prendere max 1 candidato, così che ogni anchor abbia 1 match

# Estrae la lista univoca di tutti i candidati presenti nel dataset
all_candidates = set()
for candidates in matching_data.values():
    all_candidates.update(candidates.keys())

# Vincolo 2: Ogni candidato può essere associato al massimo a un anchor (somma per colonna <= 1)
for candidate in all_candidates:
    candidate_terms = []
    for anchor in matching_data:
        if candidate in matching_data[anchor]:
            candidate_terms.append(x[anchor][candidate])
    prob += pulp.lpSum(candidate_terms) <= 1

# Risolve il programma lineare (da i match migliori a ciascun anchor minimizzando distanza e massimizzando numero di match)
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# Estrae i risultati ottimali e calcola le metriche
final_matches = {}
total_distance = 0.0
match_count = 0

for anchor in matching_data:
    for candidate in matching_data[anchor]:
        if pulp.value(x[anchor][candidate]) == 1:
            final_matches[anchor] = {
                "candidate": candidate,
                "wmd": matching_data[anchor][candidate],
            }
            total_distance += matching_data[anchor][candidate]
            match_count += 1

# Stampa l'analisi semplice dei risultati dell'ottimizzazione
print("--- Risultati dell'Ottimizzazione ---")
print(f"Stato del Solver: {pulp.LpStatus[prob.status]}")
print(f"Numero totale di Anchor matchati: {match_count}")
print(f"Distanza WMD totale combinata: {total_distance:.4f}")
if match_count > 0:
    print(f"Distanza WMD media: {(total_distance / match_count):.4f}\n")



output_path = "data/interim/matching_dict_optimized2.json"


with open(output_path, "w") as f:
    json.dump(final_matches, f, indent=4)


print(f"Salvataggio completato! File creato in: {output_path}")