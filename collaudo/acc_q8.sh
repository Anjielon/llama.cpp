#!/usr/bin/env bash
# MTP acceptance of the public engine (thor-pub/build-cpu, no Vulkan library: cannot touch the GPU), 2/11/2026.
# llama-server --spec-type draft-mtp --spec-draft-n-max 1 (or SPEC=none) on CPU, same 5 prompts as velocita_1101/mtp, temp 0.
# Saves every response (with the generated token ids) for the offline capture of the draft.
#   accettazione_cpu_pub.sh MODEL.gguf OUT_DIR [N_PREDICT] [N_PROMPTS]
set -euo pipefail
M="$1"; D="$2"; N="${3:-128}"; K="${4:-5}"
B=/mnt/models/build/thor-pub-q8/build-cpu/bin; P="${PORTA:-11491}"
# SPEC=none: plain greedy text (faster: used to build the capture sequences); default draft-mtp n=1
F="--spec-type draft-mtp --spec-draft-n-max 1"; [ "${SPEC:-mtp}" = none ] && F="--spec-type none"
mkdir -p "$D"
PROMPTS=("Spiega in modo semplice come funziona la fotosintesi clorofilliana, con un esempio." "Write a Python function that parses a CSV file and returns the rows as dictionaries, then explain its complexity." "Riassumi le cause della prima guerra mondiale in cinque punti." "A train leaves at 9:15 and travels 240 km at 80 km/h. At what time does it arrive? Show the steps." "Scrivi una breve storia per bambini su un drago che ha paura del fuoco.")
"$B/llama-server" -m "$M" -ngl 0 -c 2048 -np 1 -t 2 --host 127.0.0.1 --port "$P" \
    $F --no-repack --no-webui > "$D/server.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
for i in $(seq 1 300); do curl -s "http://127.0.0.1:$P/health" | grep -q '"ok"' && break; sleep 2; done
curl -s "http://127.0.0.1:$P/health" | grep -q '"ok"' || { echo "ERRORE: server non pronto" >&2; exit 1; }
k=0
for pr in "${PROMPTS[@]:0:$K}"; do
    body=$(python3 -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'n_predict':int(sys.argv[2]),'temperature':0,'seed':1,'cache_prompt':False,'return_tokens':True}))" "$pr" "$N")
    curl -s "http://127.0.0.1:$P/completion" -H 'Content-Type: application/json' -d "$body" > "$D/risp_$k.json"
    tok=$(python3 -c "import json,sys;print(json.dumps({'content':sys.argv[1],'add_special':True}))" "$pr")
    curl -s "http://127.0.0.1:$P/tokenize" -H 'Content-Type: application/json' -d "$tok" > "$D/prompt_$k.json"
    k=$((k+1))
done
python3 - "$D" "$K" <<'EOF'
import json, sys
d, k = sys.argv[1], int(sys.argv[2])
tn = ta = tp = 0; tps = []
for i in range(k):
    t = json.load(open(f"{d}/risp_{i}.json"))["timings"]
    tn += t.get("draft_n", 0); ta += t.get("draft_n_accepted", 0); tp += t.get("predicted_n", 0)
    tps.append(t.get("predicted_per_second", 0))
    print(i, t.get("predicted_n"), t.get("draft_n"), t.get("draft_n_accepted"), round(t.get("predicted_per_second", 0), 2))
r = {"token": tp, "bozze": tn, "accettate": ta, "accettazione": ta / max(tn, 1), "tps_medio": sum(tps) / len(tps)}
print(json.dumps(r)); json.dump(r, open(f"{d}/riepilogo.json", "w"), indent=1)
EOF
