# Stage M transform source audit

- Config SHA256: `91a360fe9087f413fccaeaa6cf56874e980773aa7f584b3bf5b2917e8a677206`
- Normalizer SHA256: `1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001`
- Channel-parameter SHA256: `917dc105eb8e1a2c2e4401e249b2e31aab595152d35f2471df263a7c426c6ddb`
- The actual source order normalizes before applying tanh softclip.
- Evaluation bounds clamp applies only to rho/press model predictions.

| stage | function | input | output | invertible | lossy |
| --- | --- | --- | --- | --- | --- |
| forward nonlinear | PaperPreprocessor._transform_values_inplace / encode_tensor | raw physical | signed-log, positive-log, or linear representation | True | False |
| robust normalization | (transformed-median)/scale | nonlinear representation | unbounded robust z | True | False |
| forward softclip | gamma*tanh(z/gamma) | unbounded robust z | bounded canonical normalized | mathematically for abs(output)<gamma | numerical saturation possible |
| inverse input clamp | clamp(encoded,-0.99*gamma,0.99*gamma) | canonical normalized | clamped normalized | False | True |
| inverse softclip | gamma*atanh(encoded/gamma) | clamped normalized | robust z | True | False |
| inverse robust normalization | z*scale+median | robust z | nonlinear representation | True | False |
| inverse nonlinearity and dtype guard | PaperPreprocessor.decode_tensor | nonlinear representation | physical | before dtype guard | only if dtype guard hits |
| evaluation-only bounds clamp | PaperPhysicalBounds.clamp_normalized | model normalized prediction | rho/press bounded normalized prediction | False | True |
