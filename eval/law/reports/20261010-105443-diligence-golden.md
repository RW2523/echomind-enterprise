# EchoMind Law due-diligence evaluation — golden

- Commit: `430853b+changes`
- Rules: `diligence-s3.7`
- Run: 20261010-105443

| Measure | Result |
|---|---|
| rooms | 3 |
| missing lists right | 3/3 |
| issues found | 18/18 |
| extra issues | 0 |
| controls clean | 1/1 |
| all preliminary | True |
| quotes at offsets | True |
| ms per room | 27 |

## G1-share-sale-room (28 ms)

- Missing: right (got corp-captable, corp-sha, disputes, insurance, ip, people; expected corp-captable, corp-sha, disputes, insurance, ip, people)
- Issues: 8/8 found

## G2-services-company-room (29 ms)

- Missing: right (got corp-articles, corp-minutes, corp-sha, disputes, ip, property; expected corp-articles, corp-minutes, corp-sha, disputes, ip, property)
- Issues: 10/10 found

## G3-corporate-only-control (23 ms, control)

- Missing: right (got contracts, corp-sha, disputes, ip, people, property; expected contracts, corp-sha, disputes, ip, people, property)
- Issues: 0/0 found
