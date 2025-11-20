-- ============================================================================
-- View: teamblue.dwh.v_deferred_invoices
-- ============================================================================
-- Purpose: Compare DWH deferred revenue with teamblue.silver.deferred_revenue
-- Granularity: INVOICE × DEFERRAL PERIOD (one row per invoice per period)
-- Limitation: NO line-item detail (invoice-level amounts only)
-- ============================================================================
-- Created: 2025-11-18
-- Last Modified: 2025-11-18
-- ============================================================================

CREATE OR REPLACE VIEW teamblue.dwh.v_deferred_invoices AS
SELECT
    -- ========================================================================
    -- Geographic and organizational dimensions
    -- ========================================================================
    geo.GEOGRAPHY_REGION_NAME as region,
    inv_src.BK_ACCOUNTING_ENTITY as entity,
    inv_src.BK_ACCOUNTING_SUB_ENTITY as sub_entity,
    
    -- ========================================================================
    -- Invoice classification and source
    -- ========================================================================
    CAST(fi.FK_INVOICE_TYPE AS STRING) as doc_type,
    inv_src.BK_INVOICE_SOURCE_NAME as source_file_name,
    CURRENT_TIMESTAMP() as workato_log_timestamp,
    
    -- ========================================================================
    -- Invoice identifiers
    -- ========================================================================
    CAST(fi.PK_INVOICES AS STRING) as invoice_pk,
    fi.BK_INVOICE_CODE as invoice_id,
    CAST(fi.PK_INVOICES AS BIGINT) as internal_id,
    fi.BK_INVOICE_CODE as billing_system_id,
    
    -- ========================================================================
    -- Transaction dates
    -- ========================================================================
    CAST(d_invoice.BK_DATE AS DATE) as transaction_date,
    
    -- ========================================================================
    -- Customer information
    -- ========================================================================
    cust.BK_CUSTOMER_CODE as customer_id,
    cust.CUSTOMER_NAME as billing_system_customer_id,
    
    -- ========================================================================
    -- Brand information
    -- ========================================================================
    brand.BUDGET_BRAND_NAME as brand_name,
    
    -- ========================================================================
    -- Currency and exchange rates
    -- ========================================================================
    curr.BK_CURRENCY_CODE as currency_code,
    CAST(1.0 AS DECIMAL(12,5)) as exchange_rate,
    CAST(1.0 AS DECIMAL(12,5)) as exchange_rate_to_eur,
    
    -- ========================================================================
    -- Due date
    -- ========================================================================
    CAST(d_order.BK_DATE AS DATE) as due_date,
    
    -- ========================================================================
    -- Product/Terms information
    -- ========================================================================
    prod.PRODUCT_NAME as terms,
    fi.DD_DESCRIPTION as memo,
    
    -- ========================================================================
    -- Additional reference fields (not available in dwh)
    -- ========================================================================
    CAST(NULL AS STRING) as other_ref_num,
    CAST(NULL AS BOOLEAN) as to_be_emailed,
    
    -- ========================================================================
    -- Billing address information (from customers dimension)
    -- ========================================================================
    cust.ADDRESS as billing_address_1,
    CAST(NULL AS STRING) as billing_address_2,
    CAST(NULL AS STRING) as billing_address_3,
    cust.CUSTOMER_NAME as addressee,
    cust.CUSTOMER_CITY as city,
    cust.CUSTOMER_COUNTRY as country,
    CAST(NULL AS STRING) as state,
    cust.POSTAL_CODE as zip,
    
    -- ========================================================================
    -- Record source
    -- ========================================================================
    'DWH' as record_source,
    
    -- ========================================================================
    -- Item/Line information (NOT AVAILABLE - invoice level only in dwh)
    -- ========================================================================
    CAST(NULL AS STRING) as item_id,
    CAST(NULL AS STRING) as billing_system_item_id,
    CAST(NULL AS STRING) as item_line_number,
    CAST(NULL AS STRING) as item_department_internal_id,
    CAST(NULL AS STRING) as item_department,
    fi.DD_DESCRIPTION as item_description,
    CAST(NULL AS STRING) as item_class_internal_id,
    
    -- ========================================================================
    -- Invoice-level amounts (NOT line-item level)
    -- ========================================================================
    CAST(fi.M_INVOICE_QUANTITY AS DECIMAL(12,2)) as item_quantity,
    CAST(fi.M_INVOICE_UNITPRICE AS DECIMAL(12,2)) as item_rate,
    CAST(fi.M_INVOICE_AMOUNT AS DECIMAL(12,2)) as item_amount,
    
    -- ========================================================================
    -- Tax information (not available at detail level in dwh)
    -- ========================================================================
    CAST(NULL AS STRING) as item_tax_code,
    CAST(NULL AS DECIMAL(12,2)) as item_tax_amount,
    
    -- ========================================================================
    -- Revenue recognition dates (from invoice dates)
    -- ========================================================================
    CAST(d_start.BK_DATE AS DATE) as rev_rec_start_date,
    CAST(d_end.BK_DATE AS DATE) as rev_rec_end_date,
    CAST(d_start.BK_DATE AS DATE) as contract_start_date,
    CAST(d_end.BK_DATE AS DATE) as contract_end_date,
    
    -- ========================================================================
    -- Period information (calculated from accrual records)
    -- ========================================================================
    COUNT(*) OVER (PARTITION BY fi.PK_INVOICES) as number_of_periods,
    ROW_NUMBER() OVER (PARTITION BY fi.PK_INVOICES ORDER BY fia.FK_DATE_ACCRUED) as period,
    
    -- ========================================================================
    -- Deferral dates and amounts
    -- ========================================================================
    CAST(d_accrued.BK_DATE AS DATE) as planned_posting_date,
    CAST(d_accrued.BK_DATE AS DATE) as actual_posting_date,
    
    -- Deferral amounts (use MRR as proxy for deferral amount)
    CAST(fi.M_INVOICE_MRR AS DECIMAL(12,2)) as planned_deferral_amount,
    CAST(fi.M_INVOICE_MRR AS DECIMAL(12,2)) as actual_deferral_amount,
    
    -- ========================================================================
    -- Metadata timestamps
    -- ========================================================================
    fi.INSERT_DATE as _modified_timestamp,
    
    -- ========================================================================
    -- Flags
    -- ========================================================================
    CAST(TRUE AS BOOLEAN) as is_historic,
    CAST(FALSE AS BOOLEAN) as is_in_netsuite,
    CAST(FALSE AS BOOLEAN) as is_in_workato,
    'DWH' as netsuite_source,
    fi.FLG_EXCLUDE_IN_REPORTING as is_excluded,
    
    -- ========================================================================
    -- NetSuite IDs (not available in dwh)
    -- ========================================================================
    CAST(NULL AS STRING) as ns_id,
    CAST(NULL AS STRING) as ns_customer_id,
    
    -- ========================================================================
    -- Invoice posting period (derived from invoice date)
    -- ========================================================================
    CAST(DATE_TRUNC('month', d_invoice.BK_DATE) AS DATE) as invoice_posting_period,
    CAST(DATE_TRUNC('month', d_invoice.BK_DATE) AS DATE) as invoice_posting_period_start_date,
    CAST(LAST_DAY(d_invoice.BK_DATE) AS DATE) as invoice_posting_period_end_date,
    
    -- ========================================================================
    -- Item class (not available at line level)
    -- ========================================================================
    CAST(NULL AS STRING) as item_class,
    
    -- ========================================================================
    -- Brand ID (from dimension)
    -- ========================================================================
    CAST(brand.PK_BUDGET_BRAND AS INT) as brand_id,
    
    -- ========================================================================
    -- Manual posting and legacy flags
    -- ========================================================================
    CAST(FALSE AS BOOLEAN) as is_manual_posting,
    CAST(FALSE AS BOOLEAN) as is_legacy,
    CAST(NULL AS TIMESTAMP) as netsuite_created_timestamp,
    
    -- ========================================================================
    -- Additional DWH-specific fields for analysis
    -- ========================================================================
    fia.ACCRUED_TYPE as dwh_accrued_type,
    fia.FX_EFFECT_LM as dwh_fx_effect_lm,
    fia.FX_EFFECT_LTM as dwh_fx_effect_ltm,
    fia.FLG_IS_NON_ACCRUED_DUE_TO_ZERO_MRR as dwh_is_non_accrued_zero_mrr,
    
    -- Product and subscription segments
    prod_seg.PRODUCT_SEGMENT_NAME as dwh_product_segment,
    bus_seg.BUSINESS_SEGMENT_NAME as dwh_business_segment

FROM 
    -- ========================================================================
    -- Main fact table: Invoices
    -- ========================================================================
    teamblue.dwh.f_invoices fi
    
    -- ========================================================================
    -- Join with invoices_accrued for period-level detail
    -- ========================================================================
    INNER JOIN teamblue.dwh.f_invoices_accrued fia
        ON fi.PK_INVOICES = fia.PK_INVOICES
    
    -- ========================================================================
    -- Date dimensions (multiple instances for different date types)
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_date d_invoice
        ON fi.FK_DATE_INVOICE = d_invoice.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_order
        ON fi.FK_DATE_ORDER = d_order.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_start
        ON fi.FK_DATE_START = d_start.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_end
        ON fi.FK_DATE_END = d_end.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_accrued
        ON fia.FK_DATE_ACCRUED = d_accrued.PK_DATE
    
    -- ========================================================================
    -- Customer dimensions
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_customers cust
        ON fi.FK_CUSTOMERS = cust.PK_CUSTOMERS
    
    -- ========================================================================
    -- Geography dimension (from fact table directly)
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_geography geo
        ON fi.FK_GEOGRAPHY_CUSTOMER = geo.PK_GEOGRAPHY
    
    -- ========================================================================
    -- Product dimensions
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_products prod
        ON fi.FK_PRODUCTS = prod.PK_PRODUCTS
    LEFT JOIN teamblue.dwh.d_product_segment prod_seg
        ON fi.FK_PRODUCT_SEGMENT = prod_seg.PK_PRODUCT_SEGMENT
    
    -- ========================================================================
    -- Brand dimension (through product segment)
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_budget_brand brand
        ON prod_seg.BK_BUDGET_BRAND_CODE = brand.BK_BUDGET_BRAND_CODE
    
    -- ========================================================================
    -- Legal Entity dimension (through brand)
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_legal_entity legal_ent
        ON brand.FK_LEGAL_ENTITY = legal_ent.PK_LEGAL_ENTITY
    
    -- ========================================================================
    -- Currency dimension
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_currency curr
        ON fi.FK_CURRENCY = curr.PK_CURRENCY
    
    -- ========================================================================
    -- Invoice source dimension
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_invoice_source inv_src
        ON fi.FK_INVOICE_SOURCE = inv_src.PK_INVOICE_SOURCE
    
    -- ========================================================================
    -- Provider dimension
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_providers prov
        ON fi.FK_PROVIDERS = prov.PK_PROVIDERS
    
    -- ========================================================================
    -- Subscription dimension
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_subscriptions subs
        ON fi.FK_SUBSCRIPTIONS = subs.PK_SUBSCRIPTIONS
    
    -- ========================================================================
    -- Business segment dimension (from fact table)
    -- ========================================================================
    LEFT JOIN teamblue.dwh.d_business_segment bus_seg
        ON fi.FK_BUSINESS_SEGMENT = bus_seg.PK_BUSINESS_SEGMENT

WHERE
    -- ========================================================================
    -- Filter 1: Only future accrual dates (deferred revenue)
    -- ========================================================================
    CAST(d_accrued.BK_DATE AS DATE) > CURRENT_DATE()
    
    -- ========================================================================
    -- Filter 2: Only RegularMRR type (exclude KPI_LTM duplicates)
    -- ========================================================================
    AND fia.ACCRUED_TYPE = 'RegularMRR'
    
    -- ========================================================================
    -- Filter 3: Exclude records marked for exclusion
    -- ========================================================================
    AND (fi.FLG_EXCLUDE_IN_REPORTING IS NULL OR fi.FLG_EXCLUDE_IN_REPORTING = FALSE)
;

-- ============================================================================
-- USAGE NOTES
-- ============================================================================
-- 
-- GRANULARITY:
--   This view is at INVOICE × PERIOD level (one row per invoice per deferral period)
--   silver.deferred_revenue is at INVOICE × LINE_ITEM × PERIOD level
--
-- COMPARISON STRATEGY:
--   To compare with silver, aggregate line items to invoice level:
--
--   -- DWH view (already at invoice level per period)
--   SELECT invoice_id, actual_posting_date, SUM(actual_deferral_amount) as dwh_amount
--   FROM teamblue.dwh.v_deferred_invoices
--   GROUP BY invoice_id, actual_posting_date
--
--   -- Silver (aggregate line items to invoice level)
--   SELECT invoice_id, actual_posting_date, SUM(actual_deferral_amount) as silver_amount
--   FROM teamblue.silver.deferred_revenue
--   GROUP BY invoice_id, actual_posting_date
--
-- LIMITATIONS:
--   - NO line-item detail (item_line_number, item_id, etc. are NULL)
--   - Amounts are at invoice level, not split by product/line
--   - Deferral amounts use M_INVOICE_MRR as proxy
--   - Some metadata fields not available in dwh (Netsuite IDs, etc.)
--
-- SAMPLE QUERIES:
--
--   -- Count deferred invoices by brand
--   SELECT brand_name, COUNT(DISTINCT invoice_id) as invoice_count
--   FROM teamblue.dwh.v_deferred_invoices
--   GROUP BY brand_name
--   ORDER BY invoice_count DESC;
--
--   -- Total deferred revenue by month
--   SELECT 
--       DATE_TRUNC('month', actual_posting_date) as month,
--       SUM(actual_deferral_amount) as total_deferred
--   FROM teamblue.dwh.v_deferred_invoices
--   GROUP BY DATE_TRUNC('month', actual_posting_date)
--   ORDER BY month;
--
-- ============================================================================
-- END OF VIEW DEFINITION
-- ============================================================================
