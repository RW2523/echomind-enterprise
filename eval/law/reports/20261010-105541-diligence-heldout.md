# EchoMind Law due-diligence evaluation — heldout

- Commit: `05863a6`
- Rules: `diligence-s3.7`
- Run: 20261010-105541

| Measure | Result |
|---|---|
| rooms | 3 |
| missing lists right | 3/3 |
| issues found | 9/11 |
| extra issues | 0 |
| controls clean | 1/1 |
| all preliminary | True |
| quotes at offsets | True |
| ms per room | 26 |

## H1-software-sale-room (31 ms)

- Missing: right (got corp-captable, corp-minutes, insurance, people, property; expected corp-captable, corp-minutes, insurance, people, property)
- Issues: 5/7 found
- **missed** ✗ Licence Agreement - Marsh & Teague.txt: renewal ×1
- **missed** ✗ Licence Agreement - Marsh & Teague.txt: expiry ×1

## H2-property-room (22 ms)

- Missing: right (got contracts, corp-articles, corp-captable, corp-sha, disputes, insurance, ip; expected contracts, corp-articles, corp-captable, corp-sha, disputes, insurance, ip)
- Issues: 4/4 found

## H3-corporate-control (25 ms, control)

- Missing: right (got contracts, corp-captable, corp-sha, disputes, ip, people, property; expected contracts, corp-captable, corp-sha, disputes, ip, people, property)
- Issues: 0/0 found
