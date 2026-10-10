# EchoMind Law governance checks evaluation — golden

- Commit: `a5ed32c+changes`
- Rules: `diligence-s3.8`
- Run: 20261010-124259

| Measure | Result |
|---|---|
| companies | 3 |
| found | 9/9 |
| extra | 0 |
| controls clean | 1/1 |
| by kind | {'share_capital': '3/3', 'authority': '2/2', 'constitution': '3/3', 'ip_chain': '1/1'} |
| all preliminary | True |
| quotes at offsets | True |
| ms per company | 29 |

## G1-tidewater (29 ms)

- share_capital: `150 ordinary shares in the register have no approval` found
- share_capital: `B ordinary shares are not a class the articles create` found
- authority: `Signed above the signer's authority` found
- constitution: `Board quorum` found
- constitution: `Notice of board meetings` found
- ip_chain: `No IP assignment from B. Lind` found

## G2-harrow-control (29 ms, control)

- (control: nothing expected)

## G3-nair (29 ms)

- share_capital: `200 preference shares in the register have no approval` found
- authority: `Signed above the signer's authority` found
- constitution: `Number of directors` found
