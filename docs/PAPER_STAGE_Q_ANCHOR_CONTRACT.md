# Stage Q persistence-anchored output contract

Stage Q is a `no_training_output_contract_audit`. It strictly reloads the
frozen Stage O best checkpoint at epoch 23, keeps the P3 transform and all
model parameters fixed, and changes only the interpretation of the eight
normalized model outputs. No backward pass, optimizer, scheduler, checkpoint
write, transform refit, or output clipping is used.

For the frozen direct map

```text
y_direct = F_theta(concat(z_t, shells))
r_theta  = y_direct - z_t
y_alpha  = z_t + alpha * r_theta
```

the scalar grid is `[0, 0.125, 0.25, 0.5, 1]`. Alpha 0 is exact persistence,
alpha 1 is the exact Stage O direct-output contract, and only the three middle
values are eligible candidates. Anchoring acts on the eight state outputs;
the eight fixed radial shell channels remain model inputs and are neither
modified nor emitted by the contract.

All Q1--Q8 contract checks pass. Endpoint tensors are bitwise exact, the
affine and residual identities pass the frozen float32 ULP-scaled tolerance,
fixed points and alpha ordering are preserved, shell hashes are unchanged,
and the model parameter/buffer SHA256 remains
`071a764bf985bfdc674c48024bfc43b4e7906ef4d3c41ac6ff2523e2926354c1`.
The contract source contains no clamp, clip, encoder, decoder, bounds repair,
or `nan_to_num` call.

Provenance gates pass for dataset
`cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a`,
P3 statistics
`aea6d3f91e54865deeb9b2b1142c49a633ae8b67223075273e6c0b88bf541948`,
Stage O best checkpoint
`756ee7f31a9780942aa0f35dce4e5584cbeaf04ed8e9777721e7765962f04e12`,
and upstream commit
`86a8bc7812a31b42c4f7895693cf4ac11521c066`. The strict reload contains four
differential modules, zero DISCO modules, and 358,296 parameters.

The complete definitions and identity evidence are in
`outputs/paper_reduced100/stage_q/contract_definition.json` and
`contract_identity_tests.json`.
