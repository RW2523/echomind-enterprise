# EchoMind Law governance checks evaluation — golden

- Commit: `df3b8a5`
- Rules: `diligence-s3.8b`
- Run: 20261010-135259

| Measure | Result |
|---|---|
| companies | 6 |
| found | 15/15 |
| extra | 0 |
| controls clean | 2/2 |
| by kind | {'share_capital': '4/4', 'authority': '4/4', 'constitution': '5/5', 'ip_chain': '2/2'} |
| all preliminary | True |
| quotes at offsets | True |
| ms per company | 35 |

## G1-tidewater (52 ms)

- share_capital: `150 ordinary shares in the register have no approval` found
- share_capital: `B ordinary shares are not a class the articles create` found
- authority: `Signed above the signer's authority` found
- constitution: `Board quorum` found
- constitution: `Notice of board meetings` found
- ip_chain: `No IP assignment from B. Lind` found

## G2-harrow-control (25 ms, control)

- (control: nothing expected)

## G3-nair (23 ms)

- share_capital: `200 preference shares in the register have no approval` found
- authority: `Signed above the signer's authority` found
- constitution: `Number of directors` found

## G4-ardent-us (28 ms)

- share_capital: `500,000` found
- authority: `Signed above the signer's authority` found
- constitution: `Number of directors` found
- ip_chain: `No IP assignment from Luis Ortega` found

## G5-brenner-uk (54 ms)

- authority: `Signed above the signer's authority` found
- constitution: `Notice of board meetings` found

## G6-calder-control (27 ms, control)

- (control: nothing expected)
