# Stage L provenance and attribution manifest

- Status: `attribution_complete`
- Classification: `post_hoc_attribution`
- Training/backward/optimizer/scheduler: `not used`
- Project commit: `8eeb27a0d0cc925ccb681e3489008bef9b14a406`
- Upstream commit: `86a8bc7812a31b42c4f7895693cf4ac11521c066`
- HDF5 SHA256: `cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a`
- Preprocessing SHA256: `1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001`
- Pair-order SHA256: `5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52`
- Detector implementation SHA256: `3a9ea391b663df6b58a0c75942283658ee0214669d234e2c5a9594cc5f36e95d`
- Detector config SHA256: `864f62987b4a36fcfeef88bb03da33598de94b8b9e8b5b2e210d81fd8c207930`
- Detector contract SHA256: `2e0d4df70e9f074f410ea29ad806d03fa44e9b24aa95650e3305d291d996511b`
- Stage L config SHA256: `8f9c170be023da938ce6b6d1414ebc459965467080ea397a85fe98998ff34983`

## Model-only strict reload

- `fno_full/best_validation_l2`: epoch `28`, state file `a18e8711aae8071218959d34d217bb52b51a1fcc23f4e11bdcc5713dbff111e2`, strict/eval `True/True`
- `fno_full/last`: epoch `30`, state file `69e03ac30fb35cb0653218c77c4e2cc38f3d13f84571cd3bbcbae6c5affe277b`, strict/eval `True/True`
- `fno_plain/best_validation_l2`: epoch `27`, state file `923b233de1d0abfda97f99fda9142dc437dc91f5dbbc3f19e0e5d28f5e862f1b`, strict/eval `True/True`
- `fno_plain/last`: epoch `30`, state file `17983d023166aecf9e42e233c613ff3b07566445382fdf2babfb425d45ca3305`, strict/eval `True/True`
- `localno_plain/best_validation_l2`: epoch `22`, state file `f69008f91da80a4a7a6374adc253bbcdf37f58f0d142ea80f2b3f399e1f02a9f`, strict/eval `True/True`
- `localno_plain/last`: epoch `30`, state file `d8ab630075126b68a5c005ea9cb7d3ac55c17f53e826a47cd74cf04a6f826b81`, strict/eval `True/True`

All checkpoint directories were discovered by matching checkpoint sidecars to the config checksums frozen in the Stage G/K run manifests. No optimizer or scheduler was instantiated or loaded.

## Phase 2 execution

- Selected GT steps: `1, 3, 5, 10, 19`
- Selected no-GT steps: `25, 50, 75, 100`
- Frozen severe-retention rule: `< 0.5`
- Bcc3 attribution: `C. MIXED_PREPROCESSING_AND_MODEL`
- vel3 attribution: `C. MIXED_PREPROCESSING_AND_MODEL`
- Overall attribution: `3. MIXED_OVERALL`
- Undefined ratios: `136`; recorded as JSON `null` and excluded from evidence aggregation
- Maximum Parseval relative error: `6.08325654478406e-16`
- Stage K decision remains: `C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`

FNO step 3 was deterministically supplemented from strict best-checkpoint reloads on
CUDA because the frozen selected-state files did not contain that requested step.
Both recomputed step-1 tensors matched their saved CUDA tensors with relative L2
`0.0`. Evaluation used `eval()` and `torch.no_grad()`; no backward pass, optimizer,
scheduler, checkpoint write, preprocessing fit, or rollout beyond existing horizons
was performed.
