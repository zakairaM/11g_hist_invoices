# DWH vs Silver Deferred Revenue Comparison Guide

## 🎯 Purpose
This guide explains how to compare deferred revenue between `teamblue.dwh.v_deferred_invoices` and `teamblue.silver.deferred_revenue`.

## 📊 Granularity Differences

### DWH View: `v_deferred_invoices`
- **Granularity:** INVOICE × PERIOD
- **One row per:** Invoice per deferral period
- **Example:** Invoice #12345 with 12 periods = 12 rows

### Silver Table: `deferred_revenue`
- **Granularity:** INVOICE × LINE_ITEM × PERIOD  
- **One row per:** Invoice per line item per deferral period
- **Example:** Invoice #12345 with 3 line items, each 12 periods = 36 rows

## ⚠️ Key Differences

| Aspect | DWH View | Silver Table |
|--------|----------|--------------|
| **Line Items** | ❌ Not available (invoice-level only) | ✅ Full line-item detail |
| **Amounts** | Invoice totals (M_INVOICE_MRR) | Line-item specific amounts |
| **Item Details** | NULL (no item_id, item_description) | Full item metadata |
| **Deferral Calc** | Uses MRR as proxy | Actual calculated deferral amounts |
| **Periods** | From f_invoices_accrued | Calculated with rev rec logic |

## ✅ Comparison Queries

### 1. Invoice-Level Comparison (Recommended)

Aggregate both to invoice + period level:

```sql
-- DWH: Already at invoice + period level
WITH dwh_summary AS (
    SELECT 
        invoice_id,
        actual_posting_date as accrual_date,
        SUM(actual_deferral_amount) as dwh_amount,
        COUNT(*) as dwh_period_count
    FROM teamblue.dwh.v_deferred_invoices
    GROUP BY invoice_id, actual_posting_date
),

-- SILVER: Aggregate line items to invoice level
silver_summary AS (
    SELECT 
        invoice_id,
        actual_posting_date as accrual_date,
        SUM(actual_deferral_amount) as silver_amount,
        COUNT(DISTINCT item_line_number) as silver_line_count,
        COUNT(*) as silver_row_count
    FROM teamblue.silver.deferred_revenue
    GROUP BY invoice_id, actual_posting_date
)

-- Compare
SELECT 
    COALESCE(d.invoice_id, s.invoice_id) as invoice_id,
    COALESCE(d.accrual_date, s.accrual_date) as accrual_date,
    d.dwh_amount,
    s.silver_amount,
    s.silver_line_count,
    (d.dwh_amount - s.silver_amount) as difference,
    CASE 
        WHEN d.dwh_amount IS NULL THEN 'Missing in DWH'
        WHEN s.silver_amount IS NULL THEN 'Missing in Silver'
        WHEN ABS(d.dwh_amount - s.silver_amount) < 0.01 THEN 'Match'
        ELSE 'Mismatch'
    END as status
FROM dwh_summary d
FULL OUTER JOIN silver_summary s
    ON d.invoice_id = s.invoice_id 
    AND d.accrual_date = s.accrual_date
WHERE ABS(COALESCE(d.dwh_amount, 0) - COALESCE(s.silver_amount, 0)) > 0.01
ORDER BY ABS(d.dwh_amount - s.silver_amount) DESC NULLS LAST
LIMIT 100;
```

### 2. Brand-Level Summary Comparison

```sql
SELECT 
    'DWH' as source,
    brand_name,
    COUNT(DISTINCT invoice_id) as invoice_count,
    SUM(actual_deferral_amount) as total_deferred
FROM teamblue.dwh.v_deferred_invoices
GROUP BY brand_name

UNION ALL

SELECT 
    'SILVER' as source,
    brand_name,
    COUNT(DISTINCT invoice_id) as invoice_count,
    SUM(actual_deferral_amount) as total_deferred
FROM teamblue.silver.deferred_revenue
GROUP BY brand_name

ORDER BY brand_name, source;
```

### 3. Period Distribution Comparison

```sql
-- How many periods per invoice?
SELECT 
    'DWH' as source,
    number_of_periods,
    COUNT(DISTINCT invoice_id) as invoice_count
FROM teamblue.dwh.v_deferred_invoices
GROUP BY number_of_periods

UNION ALL

SELECT 
    'SILVER' as source,
    number_of_periods,
    COUNT(DISTINCT invoice_id) as invoice_count
FROM teamblue.silver.deferred_revenue
GROUP BY number_of_periods

ORDER BY source, number_of_periods;
```

### 4. Date Range Comparison

```sql
SELECT 
    'DWH' as source,
    MIN(actual_posting_date) as earliest_period,
    MAX(actual_posting_date) as latest_period,
    COUNT(DISTINCT invoice_id) as invoice_count,
    COUNT(*) as total_rows
FROM teamblue.dwh.v_deferred_invoices

UNION ALL

SELECT 
    'SILVER' as source,
    MIN(actual_posting_date) as earliest_period,
    MAX(actual_posting_date) as latest_period,
    COUNT(DISTINCT invoice_id) as invoice_count,
    COUNT(*) as total_rows
FROM teamblue.silver.deferred_revenue;
```

### 5. Specific Invoice Deep Dive

```sql
-- Pick a specific invoice to compare in detail
WITH invoice_to_check AS (
    SELECT '11R-INV-53698' as invoice_id  -- Example from your sample data
)

SELECT 
    'DWH' as source,
    d.actual_posting_date,
    d.period,
    d.actual_deferral_amount,
    d.item_description,
    d.item_line_number,
    d.brand_name
FROM teamblue.dwh.v_deferred_invoices d
WHERE d.invoice_id = (SELECT invoice_id FROM invoice_to_check)

UNION ALL

SELECT 
    'SILVER' as source,
    s.actual_posting_date,
    s.period,
    s.actual_deferral_amount,
    s.item_description,
    s.item_line_number,
    s.brand_name
FROM teamblue.silver.deferred_revenue s
WHERE s.invoice_id = (SELECT invoice_id FROM invoice_to_check)

ORDER BY actual_posting_date, period, source;
```

## 🔍 Understanding Discrepancies

### Common Reasons for Differences:

1. **Line-Item Aggregation**
   - Silver has individual line items
   - DWH has invoice totals only
   - **Fix:** Aggregate silver by invoice + period

2. **Amount Calculation Methods**
   - DWH uses M_INVOICE_MRR as deferral amount
   - Silver calculates actual deferral amounts with rev rec logic
   - **Expected:** Some variance due to different calculation methods

3. **Filtering Differences**
   - DWH filters: `ACCRUED_TYPE = 'RegularMRR'` and `FK_DATE_ACCRUED > CURRENT_DATE()`
   - Silver may have different filtering logic
   - **Check:** Ensure both use same date cutoffs

4. **Data Refresh Timing**
   - DWH and Silver may refresh at different times
   - **Check:** Compare export_timestamp fields

## 📈 Recommended Analysis Workflow

1. **Start with counts:**
   ```sql
   SELECT COUNT(*), COUNT(DISTINCT invoice_id) FROM teamblue.dwh.v_deferred_invoices;
   SELECT COUNT(*), COUNT(DISTINCT invoice_id) FROM teamblue.silver.deferred_revenue;
   ```

2. **Compare at invoice + period level** (Query #1 above)

3. **Investigate mismatches:**
   - Pick invoices with largest differences
   - Use Query #5 to compare side-by-side
   - Check if missing line items in DWH explain the gap

4. **Validate with source systems:**
   - Check f_invoices for invoice totals
   - Verify period counts in f_invoices_accrued

## 🎯 Success Criteria

**Perfect Match:** Not expected due to line-item vs invoice-level granularity

**Acceptable Match:** 
- Invoice + period totals within 1-2% variance
- All invoices present in both systems
- Period distributions similar

**Red Flags:**
- Invoices in one system but not the other
- Large amount differences (>10%) after aggregation
- Completely different period counts

## 💡 Tips

- **Always aggregate silver to invoice level** before comparing amounts
- **Use brand/entity filters** to narrow down investigations
- **Focus on recent data first** (last 3-6 months)
- **Document any known exclusions** in both systems

## 📞 Need Help?

If you find systematic differences:
1. Check ETL logic for silver.deferred_revenue creation
2. Verify f_invoices_accrued data quality
3. Consult with tb-team-data-finance channel
4. Review with data warehouse team (Milos Milenkovic)
