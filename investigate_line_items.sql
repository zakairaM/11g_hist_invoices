-- ============================================================================
-- INVESTIGATION PART 2: Find Invoice Line Items
-- ============================================================================

-- 1. CHECK BRONZE LAYER (raw data ingestion)
-- ============================================================================
SHOW TABLES IN teamblue.bronze LIKE '*invoice*';
SHOW TABLES IN teamblue.bronze LIKE '*line*';
SHOW TABLES IN teamblue.bronze LIKE '*item*';

-- 2. CHECK SILVER LAYER 
-- ============================================================================
SHOW TABLES IN teamblue.silver LIKE '*invoice*';

-- 3. CHECK FINDATA_SANDBOX (from ETL pipeline)
-- ============================================================================
SHOW TABLES IN teamblue.findata_sandbox LIKE '*invoice*';
SHOW TABLES IN teamblue.findata_sandbox LIKE '*line*';
SHOW TABLES IN teamblue.findata_sandbox LIKE '*item*';

-- 4. SAMPLE f_invoices_accrued to understand the structure
-- ============================================================================
SELECT 
    fia.*,
    d.BK_DATE as accrual_date
FROM teamblue.dwh.f_invoices_accrued fia
LEFT JOIN teamblue.dwh.d_date d ON fia.FK_DATE_ACCRUED = d.PK_DATE
WHERE fia.PK_INVOICES = 52000067880  -- The invoice with 379 rows
ORDER BY fia.FK_DATE_ACCRUED
LIMIT 50;

-- 5. Check if there's a relationship between f_invoices and line items
-- ============================================================================
-- Look for any foreign keys or patterns in f_invoices
SELECT DISTINCT 
    DD_DESCRIPTION,
    COUNT(*) as count
FROM teamblue.dwh.f_invoices
WHERE DD_DESCRIPTION IS NOT NULL
GROUP BY DD_DESCRIPTION
ORDER BY count DESC
LIMIT 20;

-- 6. Check if BK_INVOICE_LINE_CODE might indicate line items
-- ============================================================================
SELECT 
    PK_INVOICES,
    BK_INVOICE_CODE,
    BK_INVOICE_LINE_CODE,
    M_INVOICE_AMOUNT,
    M_INVOICE_QUANTITY
FROM teamblue.dwh.f_invoices
WHERE BK_INVOICE_LINE_CODE IS NOT NULL
LIMIT 20;
