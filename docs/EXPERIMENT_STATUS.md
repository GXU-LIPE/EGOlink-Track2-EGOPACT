# Experiment Status

## Final Decision

The final selected line for Track 2 is:

```text
FINAL_DECISION: V41_selected / V41-S
final_package: V52_newofficial_V41S_final_track2.zip
final_package_sha256: 1f17e3a25dfc1b68346ec1bf50e8a181ad033316f4a99246eb88cd03c09046b9
active_kitchen_init_sha256: 1cd199ca1655e595f5781dd2ec832db719062ca3e14fb9d7d0a5691fe30b4a91
valid_check: true
auto_submit: false
```

## VAL68 Generalization Check

The final decision is anchored by the VAL68 generalization check:

| Method | Joint | Result | Tool | Micro |
|---|---:|---:|---:|---:|
| V41 selected | 0.3088 | 0.4118 | 0.3088 | 0.4587 |
| V43 aggressive reference | 0.2794 | 0.3676 | 0.2794 | 0.4396 |

## Important Interpretation

Older V7-V40 variants are preserved because they contain useful diagnostic and candidate modules. They should not be described as final-ready unless a current validation report supports promotion.

Observed final-stage pattern:

- V41 selected is more stable on the deduplicated VAL68 asset.
- V43 aggressive had stronger validation_A behavior but worse VAL68 generalization.
- V55 emergency candidates did not improve over V41 on VAL68.
- The final package was generated after the latest official EgoBench sync and passed V53 compliance audit.
