# India motor policy templates (demo)

The claimable-amount estimate uses three standard templates defined in
[`insuranceai/rules/india_motor.json`](../insuranceai/rules/india_motor.json).
They are simplified interpretations of common Indian private-car policies, not
any insurer's actual wording.

> **Verification status:** the figures below were compiled from public secondary
> sources (for example PayBima and bimacalculator.com), not from the IRDAI
> circular or an insurer's policy document. Verify against the policy schedule
> before relying on them. Insurers differ; replace the JSON with an insurer's
> real terms when available.

## Templates

| Template | Own damage | Parts depreciation | Compulsory deductible | Notes |
|---|---|---|---|---|
| Third-party only | No | n/a | n/a | Covers liability to others; vehicle repair is not claimable (₹0). |
| Standard comprehensive | Yes | Applied | Yes | Typical own-damage cover. |
| Comprehensive + zero depreciation | Yes | Waived | Yes | Add-on waives parts depreciation; consumables stay excluded unless a consumables add-on is bought. |

## Rules applied

- **Metal parts**, by vehicle age: up to 6 months 0%, up to 1 year 5%, 2 years
  10%, 3 years 15%, 4 years 25%, 5 years 35%, up to 10 years 40%, above 10 years 50%.
- **Rubber, plastic, nylon, tyres, batteries:** 50%. **Fibreglass:** 30%.
  **Glass:** 0%. **Paint material:** 50%.
- Depreciation is applied to replaced parts and paint, never to labour.
- **Compulsory deductible:** ₹1,000 up to 1500cc, ₹2,000 above. A voluntary
  deductible and co-pay percentage can be added.
- **IDV cap:** the claimable range cannot exceed the insured declared value.
  A possible total loss is flagged when the repair range reaches 75% of IDV
  (an assumption, insurers set their own threshold).

## Calculation order

1. Sum parts, labour and paint ranges for the priced visible-damage items.
2. Subtract depreciation (unless zero-depreciation), then the deductible.
3. Apply co-pay, then cap at IDV. The result stays a low-high range.

Possible internal damage is never part of this calculation.

## Known gaps

- Insurer-specific exclusions, add-ons (engine protect, consumables, return to
  invoice) and plan limits beyond IDV are not modelled.
- Vehicle age, engine size and IDV default to "new, up to 1500cc, no cap" until
  the user enters them, and the UI says so.
- Real policy data from customers should replace these templates.
