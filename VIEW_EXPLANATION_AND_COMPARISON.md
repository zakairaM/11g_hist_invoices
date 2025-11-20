# Understanding v_deferred_invoices and Comparing with Silver

## 🎯 What This View Does

`teamblue.dwh.v_deferred_invoices` creates a **deferred revenue report** from the DWH dimensional model that mirrors the structure of `teamblue.silver.deferred_revenue`.

### **Simple Explanation:**
Think of it like this:
- **Invoice:** You sell something for €120, customer pays upfront
- **Deferred Revenue:** You deliver service over 12 months, so you recognize €10/month
- **This View:** Shows each of those 12 monthly periods as separate rows

---

## 📊 View Structure Explained

### **Granularity: INVOICE × PERIOD**

```
Example Invoice: INV-12345
Total Amount: €120
Contract: 12 months

View Output:
┌─────────────┬────────┬────────────────┬────────┐
│ invoice_id  │ period │ posting_date   │ amount │
├─────────────┼────────┼────────────────┼────────┤
│ INV-12345   │ 1      │ 2025-01-01     │ €10    │
│ INV-12345   │ 2      │ 2025-02-01     │ €10    │
│ INV-12345   │ 3      │ 2025-03-01     │ €10    │
│ ...         │ ...    │ ...            │ ...    │
│ INV-12345   │ 12     │ 2025-12-01     │ €10    │
└─────────────┴────────┴────────────────┴────────┘

Result: 12 rows for one invoice (one per month)
```

### **Data Source Flow:**

```
ssas_f_invoices_accrued_v2  ←── Base table (period-level data)
         ↓
    + f_invoices  ←── Invoice header details
         ↓
    + d_customers  ←── Customer info
    + d_geography  ←── Region
    + d_budget_brand  ←── Brand
    + d_products  ←── Product details
    + d_currency  ←── Currency
    + 10+ more dimensions...
         ↓
    = v_deferred_invoices  ←── Final view
```

---

## 🔑 Key Columns Explained

### **Invoice Identification:**
```sql
invoice_pk          -- Internal ID (PK_INVOICES)
invoice_id          -- Business invoice number (e.g., "11R-INV-12345")
customer_id         -- Customer code (e.g., "11R-CUS-101842")
brand_name          -- Brand (e.g., "Hypernode", "IXL")
```

### **Period Information:**
```sql
period              -- Period number (1, 2, 3... up to number_of_periods)
number_of_periods   -- Total periods for this invoice (e.g., 12)
actual_posting_date -- When revenue is recognized (e.g., 2025-01-01)
```

### **Amounts (⭐ MOST IMPORTANT):**
```sql
actual_deferral_amount  -- Amount recognized THIS period
                        -- Uses M_AMOUNT_EUR from ssas_f_invoices_accrued_v2
                        -- This is period-specific, not invoice total!
```

### **Context:**
```sql
transaction_date    -- Original invoice date
contract_start_date -- When contract started
contract_end_date   -- When contract ends
region             -- Geographic region
entity             -- Legal entity
```

---

## 🆚 DWH View vs Silver Table - The Key Difference

### **Granularity Difference (CRITICAL!):**

```
DWH View:
┌─────────────┬────────┬────────┬────────┐
│ invoice_id  │ period │ amount │ TOTALS │
├─────────────┼────────┼────────┼────────┤
│ INV-001     │ 1      │ €50    │        │
│ INV-001     │ 2      │ €50    │        │
└─────────────┴────────┴────────┴────────┘
2 rows = 1 invoice × 2 periods

Silver Table:
┌─────────────┬──────┬────────┬────────┬────────┐
│ invoice_id  │ line │ period │ amount │ TOTALS │
├─────────────┼──────┼────────┼────────┼────────┤
│ INV-001     │ 1    │ 1      │ €30    │        │
│ INV-001     │ 1    │ 2      │ €30    │        │
│ INV-001     │ 2    │ 1      │ €20    │        │
│ INV-001     │ 2    │ 2      │ €20    │        │
└─────────────┴──────┴────────┴────────┴────────┘
4 rows = 1 invoice × 2 line items × 2 periods
```

**Translation:**
- **DWH:** Invoice has €50 per period (total for all line items)
- **Silver:** Same invoice split into 2 line items (€30 + €20 = €50 per period)

---

## ✅ How to Compare: Step-by-Step

### **Step 1: High-Level Counts**

```sql
-- DWH: Count invoices and periods
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT invoice_id) as unique_invoices,
    COUNT(DISTINCT actual_posting_date) as unique_periods
FROM teamblue.dwh.v_deferred_invoices;

-- Silver: Count invoices, line items, and periods
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT invoice_id) as unique_invoices,
    COUNT(DISTINCT item_line_number) as total_line_items,
    COUNT(DISTINCT actual_posting_date) as unique_periods
FROM teamblue.silver.deferred_revenue;
```

**What to expect:**
- Silver will have MORE rows (line items multiply the count)
- Both should have SIMILAR number of unique invoices
- Both should have SIMILAR period dates

---

### **Step 2: Aggregate Silver to Match DWH Granularity**

```sql
-- Create comparable datasets
WITH dwh_aggregated AS (
    SELECT 
        invoice_id,
        actual_posting_date,
        brand_name,
        SUM(actual_deferral_amount) as dwh_amount,
        COUNT(*) as dwh_row_count
    FROM teamblue.dwh.v_deferred_invoices
    GROUP BY invoice_id, actual_posting_date, brand_name
),
silver_aggregated AS (
    SELECT 
        invoice_id,
        actual_posting_date,
        brand_name,
        SUM(actual_deferral_amount) as silver_amount,
        COUNT(*) as silver_line_count
    FROM teamblue.silver.deferred_revenue
    GROUP BY invoice_id, actual_posting_date, brand_name
)
SELECT 
    COALESCE(d.invoice_id, s.invoice_id) as invoice_id,
    COALESCE(d.actual_posting_date, s.actual_posting_date) as posting_date,
    COALESCE(d.brand_name, s.brand_name) as brand,
    d.dwh_amount,
    s.silver_amount,
    s.silver_line_count,
    ABS(d.dwh_amount - s.silver_amount) as difference,
    CASE 
        WHEN d.dwh_amount IS NULL THEN 'Missing in DWH'
        WHEN s.silver_amount IS NULL THEN 'Missing in Silver'
        WHEN ABS(d.dwh_amount - s.silver_amount) < 0.01 THEN 'Perfect Match ✅'
        WHEN ABS(d.dwh_amount - s.silver_amount) < 1.00 THEN 'Close Match (~)'
        ELSE 'Mismatch ❌'
    END as match_status
FROM dwh_aggregated d
FULL OUTER JOIN silver_aggregated s
    ON d.invoice_id = s.invoice_id 
    AND d.actual_posting_date = s.actual_posting_date
ORDER BY difference DESC NULLS LAST
LIMIT 100;
```

---

### **Step 3: Summary Statistics**

```sql
WITH comparison AS (
    SELECT 
        d.invoice_id,
        d.actual_posting_date,
        d.actual_deferral_amount as dwh_amount,
        SUM(s.actual_deferral_amount) as silver_amount
    FROM teamblue.dwh.v_deferred_invoices d
    INNER JOIN teamblue.silver.deferred_revenue s
        ON d.invoice_id = s.invoice_id 
        AND d.actual_posting_date = s.actual_posting_date
    GROUP BY d.invoice_id, d.actual_posting_date, d.actual_deferral_amount
)
SELECT 
    COUNT(*) as total_invoice_periods,
    SUM(CASE WHEN ABS(dwh_amount - silver_amount) < 0.01 THEN 1 ELSE 0 END) as perfect_matches,
    SUM(CASE WHEN ABS(dwh_amount - silver_amount) BETWEEN 0.01 AND 1.00 THEN 1 ELSE 0 END) as close_matches,
    SUM(CASE WHEN ABS(dwh_amount - silver_amount) > 1.00 THEN 1 ELSE 0 END) as mismatches,
    ROUND(AVG(ABS(dwh_amount - silver_amount)), 2) as avg_difference,
    ROUND(SUM(dwh_amount), 2) as total_dwh,
    ROUND(SUM(silver_amount), 2) as total_silver,
    ROUND(SUM(dwh_amount) - SUM(silver_amount), 2) as total_difference
FROM comparison;
```

---

### **Step 4: Find Specific Discrepancies**

```sql
-- Invoices in DWH but not in Silver
SELECT DISTINCT invoice_id, brand_name, transaction_date
FROM teamblue.dwh.v_deferred_invoices
WHERE invoice_id NOT IN (SELECT DISTINCT invoice_id FROM teamblue.silver.deferred_revenue)
LIMIT 50;

-- Invoices in Silver but not in DWH
SELECT DISTINCT invoice_id, brand_name, transaction_date
FROM teamblue.silver.deferred_revenue
WHERE invoice_id NOT IN (SELECT DISTINCT invoice_id FROM teamblue.dwh.v_deferred_invoices)
LIMIT 50;
```

---

## 📈 Example Comparison Output

### **Expected Results:**

```
Summary Statistics:
├─ Total invoice-periods: 1,234,567
├─ Perfect matches: 1,150,000 (93%) ✅
├─ Close matches: 60,000 (5%) ~
├─ Mismatches: 24,567 (2%) ❌
├─ Avg difference: €0.15
├─ Total DWH: €45,678,912.50
└─ Total Silver: €45,680,123.25
    └─ Difference: €1,210.75 (0.003%)
```

### **Good Results Look Like:**
- ✅ 90%+ perfect or close matches
- ✅ Total amounts within 1-2% difference
- ✅ Most invoices present in both systems
- ✅ Average difference under €1

### **Red Flags:**
- ❌ Large number of missing invoices in either system
- ❌ Total amounts differ by >10%
- ❌ Systematic patterns in mismatches (e.g., all one brand)

---

## 🔍 Understanding Differences

### **1. Rounding Differences (Expected)**
```
DWH:    €10.00
Silver: €10.01
Reason: Different decimal precision in calculations
Status: ✅ Acceptable
```

### **2. Period Split Logic (Expected)**
```
DWH:    €100 split into 12 periods = €8.33 per period
Silver: €100 split differently, maybe €8.34 first month
Reason: Different rounding/proration logic
Status: ✅ Acceptable if total matches
```

### **3. Line Item Aggregation (Expected)**
```
DWH:    1 row per period with total €50
Silver: 3 rows (3 line items) with €50 total
Reason: Granularity difference
Status: ✅ Expected, amounts should match after aggregation
```

### **4. Missing Invoices (Investigate)**
```
Invoice in DWH but not Silver:
Reason: Maybe filtered out in silver logic?
Status: ⚠️ Investigate

Invoice in Silver but not DWH:
Reason: Maybe not loaded to DWH yet?
Status: ⚠️ Investigate
```

### **5. Large Amount Differences (Investigate)**
```
DWH:    €1,000
Silver: €500
Reason: Could be actual data issue
Status: ❌ Investigate immediately
```

---

## 💡 Pro Tips for Analysis

### **Filter to Recent Data First:**
```sql
-- Focus on last 3 months
WHERE actual_posting_date >= DATE_SUB(CURRENT_DATE(), 90)
```

### **Group by Brand:**
```sql
-- See which brands have issues
GROUP BY brand_name
ORDER BY SUM(ABS(dwh_amount - silver_amount)) DESC
```

### **Check Specific Invoice Deep Dive:**
```sql
-- Pick one problem invoice and investigate
WHERE invoice_id = 'INV-PROBLEM-123'
```

### **Look at Period Patterns:**
```sql
-- Are certain months problematic?
SELECT 
    DATE_TRUNC('month', actual_posting_date) as month,
    COUNT(*) as mismatches
FROM comparison
WHERE ABS(dwh_amount - silver_amount) > 1.00
GROUP BY month
ORDER BY month;
```

---

## 📋 Comparison Checklist

- [ ] Run Step 1 (high-level counts) - record numbers
- [ ] Run Step 2 (aggregated comparison) - check match rate
- [ ] Run Step 3 (summary statistics) - document results
- [ ] Run Step 4 (find discrepancies) - investigate missing invoices
- [ ] Filter to recent data (last 3-6 months)
- [ ] Check by brand (are certain brands problematic?)
- [ ] Document systematic differences
- [ ] Investigate large mismatches (>€10 difference)
- [ ] Verify with source systems if needed
- [ ] Share findings with data team

---

## 🎯 Success Criteria

**✅ Comparison is successful when:**
1. 90%+ of invoice-periods match within €1
2. Total amounts within 1-2% difference
3. Any systematic patterns explained
4. Missing invoices documented and understood
5. Large discrepancies (>€10) investigated

**📊 Document your findings:**
- Match rate: ____%
- Total amount difference: €_____ (_____%)
- Number of missing invoices: _____
- Key issues found: _____

---

## 🚀 What's Next?

After running the comparison:

1. **If results are good (>90% match):**
   - ✅ DWH view is validated!
   - Use it for reporting/analysis
   - Schedule regular comparison checks

2. **If results show issues (<80% match):**
   - Investigate systematic patterns
   - Check data quality in ssas_f_invoices_accrued_v2
   - Compare with source systems
   - Consult with data team

3. **For ongoing monitoring:**
   - Schedule weekly/monthly comparison reports
   - Alert on new significant discrepancies
   - Document expected variances

---

## 📞 Need Help?

**For comparison questions:**
- Check if you're aggregating silver correctly (sum by invoice + period)
- Verify date filters match (both using future periods only)
- Ensure same brands/entities being compared

**For data issues:**
- Contact: Milos Milenkovic (DWH owner)
- Channel: tb-team-data-finance
- Document: Specific invoice IDs with issues

---

**Good luck with your comparison! 🎯**
