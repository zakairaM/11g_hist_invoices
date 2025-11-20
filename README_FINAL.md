# Deferred Revenue View - Final Deliverables

## 🎉 Project Complete!

Created a comprehensive view to compare DWH deferred revenue with Silver layer.

---

## 📁 Files Created

### 1. **`create_v_deferred_invoices_FINAL.sql`** ⭐ MAIN FILE
The complete SQL view definition. Execute this in Databricks to create the view.

**What it does:**
- Creates `teamblue.dwh.v_deferred_invoices` view
- Joins `f_invoices` with `f_invoices_accrued` for period-level detail
- Includes all necessary dimension tables
- Filters for future periods only (deferred revenue)
- Matches silver.deferred_revenue structure where possible

**Execute:**
```sql
-- Copy and paste the entire contents of create_v_deferred_invoices_FINAL.sql
-- into Databricks SQL Editor and run
```

### 2. **`COMPARISON_GUIDE.md`** 📊 HOW TO USE
Detailed guide on comparing DWH vs Silver deferred revenue.

**Includes:**
- Granularity differences explanation
- 5 ready-to-use comparison queries
- Troubleshooting tips
- Expected variance explanations

### 3. **`VIEW_DOCUMENTATION.md`** 📖 REFERENCE
Original documentation (now superseded by COMPARISON_GUIDE.md, but still useful for understanding view structure)

---

## 🎯 Key Findings from Investigation

### ✅ What We Discovered:

1. **DWH Has Period-Level Data!**
   - `f_invoices_accrued` contains multiple rows per invoice (one per deferral period)
   - Example: Invoice `52000067880` has 379 deferral periods!
   - This was the breakthrough moment 🎉

2. **Granularity Difference:**
   ```
   DWH:    INVOICE × PERIOD (no line items)
   Silver: INVOICE × LINE_ITEM × PERIOD
   ```

3. **Missing from DWH:**
   - Individual line items (item_line_number, item_id, item_description)
   - Line-level amounts (item_quantity, item_rate per line)
   - Some calculated deferral fields

4. **Available in DWH:**
   - Invoice-level amounts (M_INVOICE_MRR, M_INVOICE_AMOUNT)
   - All deferral periods
   - Full dimension context (customer, brand, geography, etc.)

---

## 🚀 Quick Start

### Step 1: Create the View
```sql
-- Execute create_v_deferred_invoices_FINAL.sql in Databricks
```

### Step 2: Verify It Works
```sql
-- Check row count
SELECT COUNT(*), COUNT(DISTINCT invoice_id) 
FROM teamblue.dwh.v_deferred_invoices;

-- Sample data
SELECT * FROM teamblue.dwh.v_deferred_invoices LIMIT 10;
```

### Step 3: Run Basic Comparison
```sql
-- Invoice + period level comparison
WITH dwh_summary AS (
    SELECT 
        invoice_id,
        actual_posting_date,
        SUM(actual_deferral_amount) as dwh_amount
    FROM teamblue.dwh.v_deferred_invoices
    GROUP BY invoice_id, actual_posting_date
),
silver_summary AS (
    SELECT 
        invoice_id,
        actual_posting_date,
        SUM(actual_deferral_amount) as silver_amount
    FROM teamblue.silver.deferred_revenue
    GROUP BY invoice_id, actual_posting_date
)
SELECT 
    COUNT(*) as total_invoice_periods,
    SUM(CASE WHEN ABS(dwh_amount - silver_amount) < 0.01 THEN 1 ELSE 0 END) as matches,
    SUM(CASE WHEN ABS(dwh_amount - silver_amount) >= 0.01 THEN 1 ELSE 0 END) as mismatches
FROM dwh_summary d
INNER JOIN silver_summary s
    ON d.invoice_id = s.invoice_id 
    AND d.actual_posting_date = s.actual_posting_date;
```

---

## 📊 View Specifications

### Granularity
- **One row per:** Invoice per deferral period
- **Time range:** Future periods only (deferred revenue)
- **Filter:** RegularMRR type only (excludes KPI_LTM duplicates)

### Key Columns

| Column | Source | Description |
|--------|--------|-------------|
| `invoice_id` | f_invoices.BK_INVOICE_CODE | Invoice identifier |
| `actual_posting_date` | f_invoices_accrued.FK_DATE_ACCRUED | Deferral period date |
| `actual_deferral_amount` | f_invoices.M_INVOICE_MRR | MRR amount (invoice level) |
| `number_of_periods` | Calculated | Total periods for invoice |
| `period` | Calculated | Period number (1, 2, 3...) |
| `brand_name` | d_budget_brand | Brand name |
| `customer_id` | d_customers | Customer code |

### Filters Applied
```sql
WHERE 
    CAST(d_accrued.BK_DATE AS DATE) > CURRENT_DATE()  -- Future only
    AND fia.ACCRUED_TYPE = 'RegularMRR'                -- Skip duplicates
    AND (fi.FLG_EXCLUDE_IN_REPORTING IS NULL 
         OR fi.FLG_EXCLUDE_IN_REPORTING = FALSE)       -- Not excluded
```

---

## ⚠️ Important Limitations

### What This View CANNOT Do:

1. **Show line-item detail**
   - No item_line_number, item_id, item_description
   - Amounts are at invoice level, not split by line item

2. **Match silver row-for-row**
   - Silver has more rows (line items × periods)
   - DWH has fewer rows (invoice × periods)

3. **Provide exact deferral calculations**
   - Uses MRR as proxy for deferral amount
   - Silver has actual calculated deferral amounts

### What to Expect:

✅ **Good for:**
- Invoice-level deferred revenue comparison
- Period distribution analysis
- Brand/customer-level aggregates
- Identifying missing invoices

❌ **Not good for:**
- Line-item level reconciliation
- Product-specific deferral analysis
- Exact amount matching (some variance expected)

---

## 🔍 Investigation Scripts (Optional)

Additional investigation scripts created during discovery:

- `investigate_deferred_revenue_sources.sql` - Schema exploration
- `investigate_line_items.sql` - Line item search queries

These were used during investigation and can be deleted or kept for reference.

---

## 📞 Support & Next Steps

### If You Need Help:

1. **View creation issues:**
   - Check that all dimension tables exist
   - Verify naming conventions match your environment
   - Review linter errors

2. **Comparison discrepancies:**
   - Use COMPARISON_GUIDE.md queries
   - Aggregate silver to invoice level first
   - Check for missing invoices in either system

3. **Data quality concerns:**
   - Contact: Milos Milenkovic (dwh owner)
   - Channel: tb-team-data-finance
   - Check f_invoices_accrued data completeness

### Recommended Next Steps:

1. ✅ Create the view in production
2. ✅ Run comparison queries (COMPARISON_GUIDE.md)
3. ✅ Document any systematic differences found
4. ✅ Schedule regular comparison reports
5. ✅ Monitor for new deferred revenue patterns

---

## 📝 Technical Notes

### Column Name Fixes Applied:
- `GEOGRAPHY_REGION_NAME` (not REGION_NAME)
- `BUDGET_BRAND_NAME` (not BRAND_NAME)
- `BK_CURRENCY_CODE` (not CURRENCY_CODE)
- `BK_CUSTOMER_CODE` (not CUSTOMER_CODE)
- `BK_DATE` (DATE type, needs CAST for comparisons)

### Join Strategy:
- All dimension joins use LEFT JOIN (preserve all invoices)
- Brand accessed through product_segment (no direct FK in f_invoices)
- Geography accessed directly from f_invoices (not through customer)
- Business segment accessed directly from f_invoices

### Performance Considerations:
- View joins 15+ tables
- Filters reduce result set significantly
- Consider materialized view for production use
- Index on FK_DATE_ACCRUED recommended

---

## ✅ Success Metrics

**View Created Successfully When:**
- ✅ No SQL errors
- ✅ Returns rows (check COUNT(*))
- ✅ All key columns populated
- ✅ Date ranges make sense

**Comparison Successful When:**
- ✅ Invoice counts similar (within 5%)
- ✅ Total amounts comparable (within 10% after aggregation)
- ✅ All major brands represented
- ✅ Period distributions logical

---

## 🎓 Lessons Learned

1. **DWH structure is different from expected:**
   - Periods ARE stored (f_invoices_accrued)
   - Line items are NOT stored (invoice-level only)

2. **Multiple rows per invoice in f_invoices_accrued:**
   - One row per period (not just one row total)
   - Includes both RegularMRR and KPI_LTM types

3. **Column naming patterns:**
   - Dimension attributes use full names (e.g., GEOGRAPHY_REGION_NAME)
   - Business keys use BK_ prefix
   - Foreign keys use FK_ prefix

---

## 📅 Project Timeline

- **Started:** Investigation of dwh structure
- **Discovery:** f_invoices_accrued has period-level data
- **Solution:** Invoice × period level view (not line-item level)
- **Completed:** View created with comprehensive comparison guide

---

## 🏆 Final Deliverable

**View Name:** `teamblue.dwh.v_deferred_invoices`

**Purpose:** Compare DWH deferred revenue with Silver layer at invoice × period granularity

**Status:** ✅ Ready for deployment

**Files to Use:**
1. `create_v_deferred_invoices_FINAL.sql` - Execute this
2. `COMPARISON_GUIDE.md` - Use these queries
3. This README - Reference guide

---

**Questions? Issues? Let me know!** 🚀
