# EchoMind Law due-diligence evaluation — heldout

- Commit: `a710434`
- Rules: `diligence-s3.7b`
- Run: 20261010-105848

| Measure | Result |
|---|---|
| rooms | 2 |
| missing lists right | 2/2 |
| issues found | 5/6 |
| extra issues | 0 |
| controls clean | 1/1 |
| all preliminary | True |
| quotes at offsets | True |
| ms per room | 24 |

## I1-distribution-company-room (26 ms)

- Missing: right (got corp-articles, corp-captable, corp-minutes, corp-sha, disputes, insurance, ip, people; expected corp-articles, corp-captable, corp-minutes, corp-sha, disputes, insurance, ip, people)
- Issues: 5/6 found
- **missed** ✗ Distribution Agreement - Brenner Medical.txt: termination ×1

## I2-corporate-control (22 ms, control)

- Missing: right (got contracts, corp-captable, corp-sha, disputes, insurance, ip, people, property; expected contracts, corp-captable, corp-sha, disputes, insurance, ip, people, property)
- Issues: 0/0 found
