# V53 Final Compliance Audit Summary

The V53 audit was a read-only compliance check of the final package:

```text
/home/data-gxu/acm/egolink2026-main/code/track2/codex/final_submissions/V52_newofficial_V41S_final/V52_newofficial_V41S_final_track2.zip
```

## Verdict

```text
FINAL_ZIP_COMPLIANT=true
OFFICIAL_STRUCTURE_OK=true
JSON_PARSE_OK=true
ZIP_PATHS_OK=true
SHA256_MATCH=true
NO_BLANK_LINE=true
NO_DUPLICATE_ID=true
NO_MISSING_ID=true
NO_FORBIDDEN_FIELDS=true
NO_HIDDEN_METADATA=true
NO_GT_LABEL_LEAK=true
PDF_PRESENT=true
OLD_FINAL_NOT_OVERWRITTEN=true
AUTO_SUBMIT=false
```

## JSON Counts

| File | Rows | Expected | Parse OK | Blank Lines | Duplicate IDs | Missing IDs | Forbidden Fields |
|---|---:|---:|---|---:|---:|---:|---:|
| `retail6_easy.json` | 49 | 49 | true | 0 | 0 | 0 | 0 |
| `retail10_easy.json` | 63 | 63 | true | 0 | 0 | 0 | 0 |
| `kitchen4_easy.json` | 50 | 50 | true | 0 | 0 | 0 | 0 |
| `restaurant5_easy.json` | 50 | 50 | true | 0 | 0 | 0 | 0 |
| `order2_easy.json` | 97 | 97 | true | 0 | 0 | 0 | 0 |

## Checksums

Final zip:

```text
1f17e3a25dfc1b68346ec1bf50e8a181ad033316f4a99246eb88cd03c09046b9
```

Latest official `kitchen_init.py`:

```text
1cd199ca1655e595f5781dd2ec832db719062ca3e14fb9d7d0a5691fe30b4a91
```

## Audit Artifacts

Remote audit directory:

```text
/home/data-gxu/acm/egolink2026-main/code/track2/codex/outputs/V53_final_submission_compliance_audit_20260622_172739
```

Local audit archive:

```text
C:\Users\Administrator\Desktop\看\VLN-track2__V53-final-compliance-audit__completed-20260622
```
