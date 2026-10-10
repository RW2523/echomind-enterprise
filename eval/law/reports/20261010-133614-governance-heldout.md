# EchoMind Law governance checks evaluation — heldout

- Commit: `3446d1e`
- Rules: `diligence-s3.8b`
- Run: 20261010-133614

| Measure | Result |
|---|---|
| companies | 3 |
| found | 4/4 |
| extra | 0 |
| controls clean | 1/1 |
| by kind | {'share_capital': '2/2', 'authority': '1/1', 'constitution': '1/1'} |
| all preliminary | True |
| quotes at offsets | True |
| ms per company | 31 |

## U1-mercer-uk (33 ms)

- share_capital: `100 ordinary shares in the register have no approval` found
- authority: `Signed above the signer's authority` found

## U2-pike-us (29 ms)

- share_capital: `Series A preferred stock is not a class` found
- constitution: `Number of directors` found

## U3-orrin-control (32 ms, control)

- (control: nothing expected)
