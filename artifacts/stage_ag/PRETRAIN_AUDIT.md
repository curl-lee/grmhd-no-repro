# Stage AG Pretrain Audit

`STAGE_AG_PRETRAIN_READY`

- dataset identity: PASS (`3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`; shape `[212, 8, 64, 64, 64]`)
- train/validation split and dropped boundary: PASS
- frozen P3 normalizer identity: PASS (`c2a36edbb44732efb857165e9a710d393cec68c6ba285521cd9bcc7b783cffa6`)
- model identity and 363480 parameters: PASS
- initial trainable SHA256: `77252855f054a199500fa779f7340b33b6a2ce546b2082f17f93666275f4c588` (exact Stage AF/AD match)
- 300-epoch pair order: PASS (`536896a6b697c89d59d1c647dbac1cdfb4ff3b26d35f0cf420d808b6dd28ff24`)
- GPU gate: PASS (`NVIDIA GeForce RTX 5070`)
- no-update forward/loss/backward: PASS; DISCO gradient `4.4907379150390625`
- hash after backward/zero-grad unchanged: PASS
- optimizer created: false; parameter update performed: false

All frozen gates passed before the first optimizer update.
