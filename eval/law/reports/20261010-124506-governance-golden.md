# EchoMind Law governance checks evaluation — golden

- Commit: `2e9c2f4+changes`
- Rules: `diligence-s3.8b`
- Run: 20261010-124506

| Measure | Result |
|---|---|
| companies | 6 |
| found | 14/15 |
| extra | 1 |
| controls clean | 2/2 |
| by kind | {'share_capital': '3/4', 'authority': '4/4', 'constitution': '5/5', 'ip_chain': '2/2'} |
| all preliminary | True |
| quotes at offsets | True |
| ms per company | 34 |

## G1-tidewater (53 ms)

- share_capital: `150 ordinary shares in the register have no approval` found
- share_capital: `B ordinary shares are not a class the articles create` found
- authority: `Signed above the signer's authority` found
- constitution: `Board quorum` found
- constitution: `Notice of board meetings` found
- ip_chain: `No IP assignment from B. Lind` found

## G2-harrow-control (25 ms, control)

- (control: nothing expected)

## G3-nair (41 ms)

- share_capital: `200 preference shares in the register have no approval` found
- authority: `Signed above the signer's authority` found
- constitution: `Number of directors` found

## G4-ardent-us (28 ms)

- share_capital: `500,000` **missed** ✗
- authority: `Signed above the signer's authority` found
- constitution: `Number of directors` found
- ip_chain: `No IP assignment from Luis Ortega` found

## G5-brenner-uk (34 ms)

- authority: `Signed above the signer's authority` found
- constitution: `Notice of board meetings` found
- not expected: share_capital: Ordinary shares are not a class the articles create (Register of members - Brenner Medical UK.txt)

## G6-calder-control (26 ms, control)

- (control: nothing expected)
