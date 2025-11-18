-- ============================================================================
-- INVESTIGATION: Finding Deferred Revenue/Invoice Line Item Data Sources
-- ============================================================================
-- This script investigates where the detailed line-item and deferral period
-- data might exist in the teamblue data warehouse
-- ============================================================================

-- ============================================================================
-- 1. CHECK ALL SCHEMAS IN TEAMBLUE CATALOG
-- ============================================================================
SHOW SCHEMAS IN teamblue;

-- ============================================================================
-- 2. CHECK ALL TABLES IN DWH SCHEMA (comprehensive)
-- ============================================================================
SHOW TABLES IN teamblue.dwh;

-- ============================================================================
-- 3. SEARCH FOR INVOICE-RELATED TABLES ACROSS ALL SCHEMAS
-- ============================================================================
-- Check bronze layer
SHOW TABLES IN teamblue LIKE '*invoice*';
SHOW TABLES IN teamblue LIKE '*line*';
SHOW TABLES IN teamblue LIKE '*item*';

-- Check for staging tables
SHOW TABLES IN teamblue LIKE 'stg_*';

-- Check for raw/landing tables
SHOW TABLES IN teamblue LIKE 'raw_*';
SHOW TABLES IN teamblue LIKE 'landing_*';

-- ============================================================================
-- 4. SEARCH FOR DEFERRAL/REVENUE RECOGNITION TABLES
-- ============================================================================
SHOW TABLES IN teamblue LIKE '*defer*';
SHOW TABLES IN teamblue LIKE '*revenue*';
SHOW TABLES IN teamblue LIKE '*accrual*';
SHOW TABLES IN teamblue LIKE '*recognition*';

-- ============================================================================
-- 5. CHECK SILVER SCHEMA (if exists)
-- ============================================================================
SHOW TABLES IN teamblue.silver;

-- ============================================================================
-- 6. CHECK IF THERE ARE ANY VIEWS IN DWH
-- ============================================================================
SHOW VIEWS IN teamblue.dwh;

-- ============================================================================
-- 7. SPECIFIC CHECKS - Look for tables that might contain line items
-- ============================================================================
-- Check if there's a fact table for invoice lines
SHOW TABLES IN teamblue.dwh LIKE '*line*';
SHOW TABLES IN teamblue.dwh LIKE '*item*';
SHOW TABLES IN teamblue.dwh LIKE '*detail*';

-- ============================================================================
-- 8. CHECK FINDATA_SANDBOX (from ETL pipeline we saw earlier)
-- ============================================================================
SHOW TABLES IN teamblue.findata_sandbox LIKE '*invoice*';

-- ============================================================================
-- 9. INVESTIGATE f_invoices_accrued FURTHER
-- ============================================================================
-- Maybe it has more detail than we thought?
SELECT COUNT(*) as total_rows FROM teamblue.dwh.f_invoices_accrued LIMIT 10;
SELECT COUNT(DISTINCT PK_INVOICES) as unique_invoices FROM teamblue.dwh.f_invoices_accrued LIMIT 10;

-- Check if there are multiple accrual records per invoice (indicating periods)
SELECT 
    PK_INVOICES,
    COUNT(*) as accrual_records
FROM teamblue.dwh.f_invoices_accrued
GROUP BY PK_INVOICES
HAVING COUNT(*) > 1
LIMIT 20;

-- ============================================================================
-- 10. CHECK IF DEFERRED_REVENUE EXISTS IN OTHER LOCATIONS
-- ============================================================================
-- Already know it exists in silver, but check for copies or staging versions
SHOW TABLES IN teamblue LIKE '%deferred%';

-- ============================================================================
-- END OF INVESTIGATION SCRIPT
-- ============================================================================
-- Run each section separately in Databricks SQL Editor
-- Document findings to determine if line-item level data exists anywhere
-- ============================================================================
