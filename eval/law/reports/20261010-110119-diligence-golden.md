# EchoMind Law due-diligence evaluation — golden

- Commit: `a710434+changes`
- Rules: `diligence-s3.7c`
- Run: 20261010-110119

| Measure | Result |
|---|---|
| rooms | 6 |
| missing lists right | 6/6 |
| issues found | 29/29 |
| extra issues | 0 |
| controls clean | 2/2 |
| all preliminary | True |
| quotes at offsets | True |
| ms per room | 29 |

## G1-share-sale-room (30 ms)

- Missing: right (got corp-captable, corp-sha, disputes, insurance, ip, people; expected corp-captable, corp-sha, disputes, insurance, ip, people)
- Issues: 8/8 found

## G2-services-company-room (32 ms)

- Missing: right (got corp-articles, corp-minutes, corp-sha, disputes, ip, property; expected corp-articles, corp-minutes, corp-sha, disputes, ip, property)
- Issues: 10/10 found

## G3-corporate-only-control (20 ms, control)

- Missing: right (got contracts, corp-sha, disputes, ip, people, property; expected contracts, corp-sha, disputes, ip, people, property)
- Issues: 0/0 found

## G4-software-sale-room (33 ms)

- Missing: right (got corp-captable, corp-minutes, insurance, people, property; expected corp-captable, corp-minutes, insurance, people, property)
- Issues: 7/7 found

## G5-property-room (24 ms)

- Missing: right (got contracts, corp-articles, corp-captable, corp-sha, disputes, insurance, ip; expected contracts, corp-articles, corp-captable, corp-sha, disputes, insurance, ip)
- Issues: 4/4 found

## G6-corporate-control (33 ms, control)

- Missing: right (got contracts, corp-captable, corp-sha, disputes, ip, people, property; expected contracts, corp-captable, corp-sha, disputes, ip, people, property)
- Issues: 0/0 found
