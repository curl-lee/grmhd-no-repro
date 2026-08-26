# Stage O direct-state versus Stage R residual-target LocalNO

- One-step normalized average: Stage O `1.04711`, Stage R `0.224673`, P3 persistence `0.180266`.
- Stage R / Stage O: `0.214565`; Stage R / persistence: `1.24634`.
- Mean residual cosine/sign agreement: `0.131921` / `0.524242`.
- Residual targeting sharply reduces direct-state normalized overshoot, but all validation pairs remain above Rout and physical ranges explode by GT step 3.
- The long rollout remains finite and positive but carries ripple/stripe flags and extreme transform-tail exposure.
- Limitation: Stage O frozen evaluation used its authorized rho/press evaluation clamp; Stage R forbids all physical output repair. Normalized and structural comparisons remain directly paired; repaired physical ranges are not treated as exact peers.
