# Same-input routing mediation pilot (independent hypothesis)

This experiment complements, and does not replace, the terminal-edge semantic-patching test. All negative semantic results remain reported.

Question: Do attention patterns collectively carry a substantial part of the fixed J's behavioral effect in Ouro? A single-head failure does not settle this distributed hypothesis.

Use the base prompts of the same eight discovery pairs. Same checkpoint/J hashes, four calls, all-token J before calls 2–4. Compare native and full-J on identical inputs. At specified sites exchange clean-run attention patterns only, recomputing alpha_source V_receiving; output and V-only transfers are separate comparison arms. Do not patch residual states, Q, K, MLPs, or J itself. Frozen source activations constitute an oracle causal test, not an autonomous algorithm.

Fixed exploratory scopes, zero-based layers, calls 2–4 jointly:
- all 48 layers;
- early 0–15, middle 16–31, late 32–47;
- prior local region 33–34.

For each scope: native plus J patterns (rescue), native plus J head outputs, native plus J values; J plus native patterns (damage); native self-pattern and self-output controls. All heads and all prompt query rows within a scope. Baselines native and full J. All scopes declared before evaluating outcomes; report every one.

Metric: first-answer-token full-vocabulary argmax, correct-name probability and logit margin. Accuracy gain recovery (rescue-native)/(J-native) only if the denominator is positive; keep absolute accuracies. Any promising site/scope must be frozen and tested on independent graphs, matched unrelated patterns/head groups where meaningful, full-name generation, and replication before paper claims.

Identity output restoration must reproduce logits exactly. Native self-pattern checks quantify changes from FP32 recomputation versus BF16 SDPA; do not attribute reconstruction changes to source patterns. Save baseline comparisons on all eight graphs, no success-based exclusions.

## Matched-prefix final-call variant (specified before its evaluation)
A complementary local test removes only J before call 4, preserving J at calls 2 and 3. Both branches therefore share the exact incoming hidden state at the final boundary. Exchange patterns/outputs/values only during call 4; the reference run retains final J. This tests local mediation on a shared prefix, not recovery from the entirely native trajectory. Retain the same five layer scopes and six arms. Report this variant separately, with baseline labeled "omit final J", never native. If it succeeds while native rescue fails, conclude compatibility depends on the earlier state trajectory, not that routing accounts for all of J's cumulative effect.
