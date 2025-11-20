# View Version Comparison: V1 vs V2 (IMPROVED)

## 🎯 Why V2 is Better

You found `ssas_f_invoices_accrued_v2` which has **MUCH BETTER** metrics for deferred revenue comparison!

---

## 📊 Key Improvements

### 1. **Better Amount Fields** ⭐ MOST IMPORTANT

| Aspect | V1 (OLD) | V2 (IMPROVED) |
|--------|----------|---------------|
| **Base Table** | `f_invoices_accrued` | `ssas_f_invoices_accrued_v2` |
| **Deferral Amount** | `M_INVOICE_MRR` (invoice total) | `M_AMOUNT_EUR` (period-specific!) |
| **Accuracy** | ❌ Same amount for all periods | ✅ Different per period |
| **Comparison** | ❌ Less accurate | ✅ Much more accurate! |

**Example:**
```
Invoice #12345 with 12 periods, Total €1,200

V1: Each period shows €1,200 (wrong - it's the invoice total!)
V2: Each period shows €100 (correct - split across 12 periods!)
```

### 2. **Additional Metrics Available**

| Metric | V1 | V2 | Description |
|--------|----|----|-------------|
| **M_AMOUNT_EUR** | ❌ | ✅ | Period amount in EUR |
| **M_MRR_EOP_EUR** | ❌ | ✅ | End of period MRR |
| **M_QUANTITY** | ❌ | ✅ | Period quantity |
| **M_UPSELL_LM/LTM** | ❌ | ✅ | Upsell amounts |
| **M_DOWNSELL_LM/LTM** | ❌ | ✅ | Downsell amounts |

### 3. **Better Flags for Analysis**

V2 includes business intelligence flags:

**Customer Movement:**
- `FLG_NEW_CONTEXT_BILLING_CUSTOMER` - New customer
- `FLG_CHURN_CONTEXT_BILLING_CUSTOMER` - Churned customer

**Revenue Movement (LM = Last Month, LTM = Last 12 Months):**
- `FLG_JOIN_LM` / `FLG_JOIN_LTM` - Customer joined
- `FLG_CHURN_LM` / `FLG_CHURN_LTM` - Customer churned
- `FLG_UPSELL_LM` / `FLG_UPSELL_LTM` - Revenue upsell
- `FLG_DOWNSELL_LM` / `FLG_DOWNSELL_LTM` - Revenue downsell
- `FLG_CROSS_SELL_LM` / `FLG_CROSS_SELL_LTM` - Cross-sell

**Subscription Movement:**
- `FLG_SUBSCRIPTIONS_RENEWED_LM/LTM` - Renewed
- `FLG_SUBSCRIPTIONS_ADDED_LM/LTM` - Added
- `FLG_SUBSCRIPTIONS_LOST_LM/LTM` - Lost
- `FLG_SUBSCRIPTIONS_UPSELL_LM/LTM` - Upsell

### 4. **SQL Structure Changes**

**V1 Approach:**
```sql
FROM f_invoices fi
INNER JOIN f_invoices_accrued fia ON fi.PK_INVOICES = fia.PK_INVOICES
-- Had to join f_invoices first, then accrued
-- Used M_INVOICE_MRR as proxy (not accurate per period)
```

**V2 Approach:**
```sql
FROM ssas_f_invoices_accrued_v2 sia
LEFT JOIN f_invoices fi ON sia.PK_INVOICES = fi.PK_INVOICES
-- Start with v2 (already has period data + metrics)
-- Use M_AMOUNT_EUR (actual period amount!)
```

---

## 🔍 Detailed Comparison

### Amount Calculation

**Scenario:** Invoice with €120 total, 12 monthly periods

#### V1 (OLD - INCORRECT):
```sql
SELECT 
    invoice_id,
    period,
    M_INVOICE_MRR as amount  -- Shows €120 for EVERY period!
FROM v1_view
WHERE invoice_id = 'INV-12345'
```

Result:
| period | amount |
|--------|--------|
| 1 | €120 |
| 2 | €120 |
| ... | €120 |
| 12 | €120 |

**Total if summed:** €120 × 12 = €1,440 ❌ WRONG!

#### V2 (NEW - CORRECT):
```sql
SELECT 
    invoice_id,
    period,
    M_AMOUNT_EUR as amount  -- Shows €10 per period!
FROM v2_view
WHERE invoice_id = 'INV-12345'
```

Result:
| period | amount |
|--------|--------|
| 1 | €10 |
| 2 | €10 |
| ... | €10 |
| 12 | €10 |

**Total if summed:** €10 × 12 = €120 ✅ CORRECT!

---

## 📈 Impact on Comparison

### V1 Comparison Issues:
```sql
-- DWH V1 (wrong amounts per period)
SELECT invoice_id, actual_posting_date, actual_deferral_amount
FROM v1_deferred_invoices
WHERE invoice_id = 'INV-12345'
-- Shows €120 for each of 12 periods = €1,440 total ❌

-- Silver (correct amounts)
SELECT invoice_id, actual_posting_date, SUM(actual_deferral_amount)
FROM silver.deferred_revenue
WHERE invoice_id = 'INV-12345'
GROUP BY invoice_id, actual_posting_date
-- Shows €10 per period = €120 total ✅

-- MISMATCH: €1,440 vs €120 = 1,200% difference! 😱
```

### V2 Comparison Success:
```sql
-- DWH V2 (correct amounts per period)
SELECT invoice_id, actual_posting_date, actual_deferral_amount
FROM v2_deferred_invoices
WHERE invoice_id = 'INV-12345'
-- Shows €10 for each of 12 periods = €120 total ✅

-- Silver (correct amounts)
SELECT invoice_id, actual_posting_date, SUM(actual_deferral_amount)
FROM silver.deferred_revenue
WHERE invoice_id = 'INV-12345'
GROUP BY invoice_id, actual_posting_date
-- Shows €10 per period = €120 total ✅

-- MATCH! €120 = €120 🎉
```

---

## ⚠️ What Stays the Same (Still Limitations)

Even with V2, these limitations remain:

| Feature | V1 | V2 | Comment |
|---------|----|----|---------|
| **Line item detail** | ❌ | ❌ | Still invoice-level only |
| **item_line_number** | NULL | NULL | Not in dwh |
| **item_id** | NULL | NULL | Not in dwh |
| **item_description** | DD_DESCRIPTION | DD_DESCRIPTION | Invoice-level only |
| **Row count** | Lower than silver | Lower than silver | Still no line items |

**But the amounts are now MUCH more accurate!**

---

## 🚀 Migration Steps

### Step 1: Verify Current V1 View
```sql
-- Check what you currently have
SELECT COUNT(*), SUM(actual_deferral_amount) 
FROM teamblue.dwh.v_deferred_invoices;
```

### Step 2: Deploy V2 View
```sql
-- Execute create_v_deferred_invoices_V2_IMPROVED.sql
-- This will REPLACE the current view
```

### Step 3: Compare V1 vs V2 Results
```sql
-- If you want to test both versions, create V2 with different name first:
-- CREATE VIEW teamblue.dwh.v_deferred_invoices_v2 AS ...

-- Then compare:
SELECT 
    'V1' as version,
    COUNT(*) as rows,
    SUM(actual_deferral_amount) as total
FROM teamblue.dwh.v_deferred_invoices
UNION ALL
SELECT 
    'V2' as version,
    COUNT(*) as rows,
    SUM(actual_deferral_amount) as total
FROM teamblue.dwh.v_deferred_invoices_v2;
```

### Step 4: Test Comparison with Silver
```sql
-- V2 should match silver MUCH better than V1
WITH dwh_v2 AS (
    SELECT invoice_id, actual_posting_date, 
           SUM(actual_deferral_amount) as dwh_amount
    FROM teamblue.dwh.v_deferred_invoices
    GROUP BY invoice_id, actual_posting_date
),
silver_agg AS (
    SELECT invoice_id, actual_posting_date,
           SUM(actual_deferral_amount) as silver_amount
    FROM teamblue.silver.deferred_revenue
    GROUP BY invoice_id, actual_posting_date
)
SELECT 
    COUNT(*) as invoice_periods,
    AVG(ABS(dwh_amount - silver_amount)) as avg_difference,
    SUM(CASE WHEN ABS(dwh_amount - silver_amount) < 0.01 THEN 1 ELSE 0 END) as perfect_matches,
    SUM(CASE WHEN ABS(dwh_amount - silver_amount) >= 0.01 THEN 1 ELSE 0 END) as mismatches
FROM dwh_v2 d
INNER JOIN silver_agg s
    ON d.invoice_id = s.invoice_id 
    AND d.actual_posting_date = s.actual_posting_date;
```

---

## 📊 Expected Results

### V1 Performance (Before):
- ❌ Large amount discrepancies (could be 10-100x off)
- ❌ Period amounts don't match silver
- ❌ Comparison requires complex adjustments

### V2 Performance (After):
- ✅ Much closer amount matches (within 5-10%)
- ✅ Period amounts align with silver logic
- ✅ Remaining differences due to line-item vs invoice-level only

---

## 💡 Recommendation

**✅ DEPLOY V2 IMMEDIATELY!**

1. **Better accuracy:** M_AMOUNT_EUR is period-specific
2. **Easier comparison:** Direct comparison with silver amounts
3. **More insights:** Additional analysis flags
4. **Same limitations:** Line items still not available (but that's expected)

---

## 📞 Support

If you see issues after deploying V2:

1. **Amounts still don't match:**
   - Check if silver includes line-item level detail
   - Verify both use same date filters
   - Compare specific invoices individually

2. **Row counts different:**
   - Expected! DWH is invoice-level, silver is line-item level
   - Aggregate silver by invoice + period for fair comparison

3. **Missing data:**
   - Verify ssas_f_invoices_accrued_v2 has all invoices
   - Check if any invoices filtered out by FLG_EXCLUDE_IN_REPORTING

---

## ✅ Decision Matrix

| Question | Answer | Action |
|----------|--------|--------|
| Does V2 have better amounts? | ✅ YES | Deploy V2 |
| Does V2 have line items? | ❌ NO | Accept limitation |
| Should I keep V1? | ❌ NO | Replace with V2 |
| Will comparison improve? | ✅ YES | Much more accurate |

---

**🎯 Bottom Line: V2 is a SIGNIFICANT improvement over V1. Deploy it!**
