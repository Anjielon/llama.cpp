#!/bin/bash
O=/mnt/models/gguf/tony-tern/MTP_1101/prova_M_testaXS_1106; C=/mnt/models/build/thor-pub-q8/collaudo
bash $C/acc_q8.sh $O/Thor-T-35B-A3B-M.gguf $C/acc_riuso 128 5 > $C/acc_riuso.log 2>&1
python3 - <<'Q' > $C/uguali.txt 2>&1
import json
u=0
for i in range(5):
    a=json.load(open(f"/mnt/models/gguf/tony-tern/MTP_1101/prova_M_testaXS_1106/acc_vecchia/risp_{i}.json")); b=json.load(open(f"/mnt/models/build/thor-pub-q8/collaudo/acc_riuso/risp_{i}.json"))
    u+= a.get("tokens")==b.get("tokens") and a["content"]==b["content"]
print("risposte identiche",u,"su 5")
Q
printf '%s\tATTIVO\tKERNEL-Q8 riuso mappe su CPU: %s; %s\n' "$(date +%Y-%m-%dT%H:%M)" "$(cat $C/uguali.txt | tail -1)" "$(tail -1 $C/acc_riuso.log | cut -c1-110)" >> /home/angelo/tony-forgia/stato/cicli.tsv
