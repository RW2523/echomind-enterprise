# EchoMind Law governance checks evaluation — heldout

- Commit: `689331f`
- Rules: `diligence-s3.8b`
- Run: 20261010-131935

| Measure | Result |
|---|---|
| companies | 3 |
| found | 4/4 |
| extra | 0 |
| controls clean | 1/1 |
| by kind | {'share_capital': '2/2', 'authority': '1/1', 'constitution': '1/1'} |
| all preliminary | True |
| quotes at offsets | True |
| ms per company | 37 |

## U1-mercer-uk (52 ms)

- share_capital: `100 ordinary shares in the register have no approval` found
- authority: `Signed above the signer's authority` found

## U2-pike-us (36 ms)

- share_capital: `Series A preferred stock is not a class` found
- constitution: `Number of directors` found

## U3-orrin-control (22 ms, control)

- (control: nothing expected)
