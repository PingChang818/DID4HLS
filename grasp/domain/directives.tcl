set_directive_loop_merge "MixColumn_AddRoundKey"
set_directive_loop_merge "AddRoundKey_InversMixColumn"
set_directive_pipeline "AddRoundKey_InversMixColumn/AddRoundKey_InversMixColumn_label0"
set_directive_pipeline "AddRoundKey_InversMixColumn/AddRoundKey_InversMixColumn_label2"
set_directive_unroll "encrypt/encrypt_label1"
set_directive_unroll "encrypt/encrypt_label2"
set_directive_unroll "encrypt/encrypt_label3"
set_directive_unroll "decrypt/decrypt_label4"
set_directive_pipeline "decrypt/decrypt_label5"
set_directive_pipeline "KeySchedule/KeySchedule_label8"
set_directive_unroll "KeySchedule/KeySchedule_label9"
set_directive_array_partition -type block -factor 8 -dim 0 "ByteSub_ShiftRow" Sbox
set_directive_array_partition -type block -factor 16 -dim 0 "ByteSub_ShiftRow" invSbox
set_directive_array_partition -type block -factor 2 -dim 0 "aes_main" statemt
set_directive_pipeline "AddRoundKey/AddRoundKey_label0"
set_directive_pipeline "MixColumn_AddRoundKey/MixColumn_AddRoundKey_label0"
set_directive_pipeline "KeySchedule/KeySchedule_label4"
set_directive_loop_flatten "KeySchedule/KeySchedule_label5"
