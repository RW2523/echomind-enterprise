# EchoMind Law due-diligence evaluation — heldout

- Commit: `689331f`
- Rules: `diligence-s3.8b`
- Run: 20261010-131931

| Measure | Result |
|---|---|
| rooms | 2 |
| missing lists right | 2/2 |
| issues found | 5/5 |
| extra issues | 0 |
| controls clean | 1/1 |
| all preliminary | True |
| quotes at offsets | True |
| ms per room | 36 |

## T1-software-company (39 ms)

- Missing: right (got corp-captable, corp-minutes, disputes, insurance, ip, property; expected corp-captable, corp-minutes, disputes, insurance, ip, property)
- Issues: 5/5 found

## T2-corporate-control (32 ms, control)

- Missing: right (got contracts, corp-captable, corp-sha, disputes, insurance, ip, people, property; expected contracts, corp-captable, corp-sha, disputes, insurance, ip, people, property)
- Issues: 0/0 found
