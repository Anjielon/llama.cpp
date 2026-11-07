#!/bin/bash
export GGML_VK_VISIBLE_DEVICES=0 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json LD_LIBRARY_PATH=/mnt/models/build/thor-pub-q8/build-vk/bin
for q in 0 1; do for o in MUL_MAT MUL_MAT_ID; do
  GGML_VK_TQ1R_Q8=$q timeout 900 /mnt/models/build/thor-pub-q8/build-vk/bin/test-backend-ops -b Vulkan0 -o $o > /mnt/models/build/thor-pub-q8/collaudo/${o}_q$q.log 2>&1; echo "$o q8=$q rc $? $(grep -cE 'OK' /mnt/models/build/thor-pub-q8/collaudo/${o}_q$q.log) ok $(grep -cE 'FAIL' /mnt/models/build/thor-pub-q8/collaudo/${o}_q$q.log) fail" >> /mnt/models/build/thor-pub-q8/collaudo/riassunto.txt
done; done
printf '%s\tATTIVO\tKERNEL-Q8 lavapipe: %s\n' "$(date +%Y-%m-%dT%H:%M)" "$(tr '\n' ';' < /mnt/models/build/thor-pub-q8/collaudo/riassunto.txt | cut -c1-240)" >> /home/angelo/tony-forgia/stato/cicli.tsv
