-- ============================================================================
-- View: teamblue.dwh.v_deferred_invoices (V2 - IMPROVED)
-- ============================================================================
-- Purpose: Compare DWH deferred revenue with silver.deferred_revenue
-- Granularity: INVOICE × PERIOD
-- Base: ssas_f_invoices_accrued_v2 (better than f_invoices_accrued)
-- Key Improvement: Uses M_AMOUNT_EUR for accurate period-level amounts
-- ============================================================================

CREATE OR REPLACE VIEW teamblue.dwh.v_deferred_invoices AS
SELECT
    -- Geographic and organizational
    geo.GEOGRAPHY_REGION_NAME as region,
    inv_src.BK_ACCOUNTING_ENTITY as entity,
    inv_src.BK_ACCOUNTING_SUB_ENTITY as sub_entity,
    
    -- Invoice classification
    CAST(sia.FK_INVOICE_TYPE AS STRING) as doc_type,
    inv_src.BK_INVOICE_SOURCE_NAME as source_file_name,
    CURRENT_TIMESTAMP() as workato_log_timestamp,
    
    -- Invoice identifiers
    CAST(sia.PK_INVOICES AS STRING) as invoice_pk,
    fi.BK_INVOICE_CODE as invoice_id,
    CAST(sia.PK_INVOICES AS BIGINT) as internal_id,
    fi.BK_INVOICE_CODE as billing_system_id,
    
    -- Dates
    CAST(d_invoice.BK_DATE AS DATE) as transaction_date,
    
    -- Customer information
    cust.BK_CUSTOMER_CODE as customer_id,
    cust.CUSTOMER_NAME as billing_system_customer_id,
    
    -- Brand
    brand.BUDGET_BRAND_NAME as brand_name,
    
    -- Currency
    curr.BK_CURRENCY_CODE as currency_code,
    CAST(1.0 AS DECIMAL(12,5)) as exchange_rate,
    CAST(1.0 AS DECIMAL(12,5)) as exchange_rate_to_eur,
    
    -- Due date
    CAST(d_order.BK_DATE AS DATE) as due_date,
    
    -- Product/Terms
    prod.PRODUCT_NAME as terms,
    fi.DD_DESCRIPTION as memo,
    
    -- Fields not available in dwh
    CAST(NULL AS STRING) as other_ref_num,
    CAST(NULL AS BOOLEAN) as to_be_emailed,
    
    -- Billing address
    cust.ADDRESS as billing_address_1,
    CAST(NULL AS STRING) as billing_address_2,
    CAST(NULL AS STRING) as billing_address_3,
    cust.CUSTOMER_NAME as addressee,
    cust.CUSTOMER_CITY as city,
    cust.CUSTOMER_COUNTRY as country,
    CAST(NULL AS STRING) as state,
    cust.POSTAL_CODE as zip,
    
    -- Source
    'DWH_V2' as record_source,
    
    -- Line item fields (not available in dwh)
    CAST(NULL AS STRING) as item_id,
    CAST(NULL AS STRING) as billing_system_item_id,
    CAST(NULL AS STRING) as item_line_number,
    CAST(NULL AS STRING) as item_department_internal_id,
    CAST(NULL AS STRING) as item_department,
    fi.DD_DESCRIPTION as item_description,
    CAST(NULL AS STRING) as item_class_internal_id,
    
    -- Period-level amounts (IMPROVED: from ssas_f_invoices_accrued_v2)
    CAST(sia.M_QUANTITY AS DECIMAL(12,2)) as item_quantity,
    CAST(sia.M_AMOUNT_EUR / NULLIF(sia.M_QUANTITY, 0) AS DECIMAL(12,2)) as item_rate,
    CAST(sia.M_AMOUNT_EUR AS DECIMAL(12,2)) as item_amount,
    
    -- Tax (not available)
    CAST(NULL AS STRING) as item_tax_code,
    CAST(NULL AS DECIMAL(12,2)) as item_tax_amount,
    
    -- Revenue recognition dates
    CAST(d_start.BK_DATE AS DATE) as rev_rec_start_date,
    CAST(d_end.BK_DATE AS DATE) as rev_rec_end_date,
    CAST(d_start.BK_DATE AS DATE) as contract_start_date,
    CAST(d_end.BK_DATE AS DATE) as contract_end_date,
    
    -- Period calculations
    COUNT(*) OVER (PARTITION BY sia.PK_INVOICES) as number_of_periods,
    ROW_NUMBER() OVER (PARTITION BY sia.PK_INVOICES ORDER BY sia.FK_DATE_ACCUED) as period,
    
    -- Deferral dates and amounts (IMPROVED: actual period amounts)
    CAST(d_accrued.BK_DATE AS DATE) as planned_posting_date,
    CAST(d_accrued.BK_DATE AS DATE) as actual_posting_date,
    CAST(sia.M_AMOUNT_EUR AS DECIMAL(12,2)) as planned_deferral_amount,
    CAST(sia.M_AMOUNT_EUR AS DECIMAL(12,2)) as actual_deferral_amount,
    
    -- Metadata
    fi.INSERT_DATE as _modified_timestamp,
    
    -- Flags
    CAST(TRUE AS BOOLEAN) as is_historic,
    CAST(FALSE AS BOOLEAN) as is_in_netsuite,
    CAST(FALSE AS BOOLEAN) as is_in_workato,
    'DWH_V2' as netsuite_source,
    fi.FLG_EXCLUDE_IN_REPORTING as is_excluded,
    
    -- NetSuite IDs (not available)
    CAST(NULL AS STRING) as ns_id,
    CAST(NULL AS STRING) as ns_customer_id,
    
    -- Invoice posting period
    CAST(DATE_TRUNC('month', d_invoice.BK_DATE) AS DATE) as invoice_posting_period,
    CAST(DATE_TRUNC('month', d_invoice.BK_DATE) AS DATE) as invoice_posting_period_start_date,
    CAST(LAST_DAY(d_invoice.BK_DATE) AS DATE) as invoice_posting_period_end_date,
    
    -- Item class (not available)
    CAST(NULL AS STRING) as item_class,
    
    -- Brand ID
    CAST(brand.PK_BUDGET_BRAND AS INT) as brand_id,
    
    -- Manual/legacy flags
    CAST(FALSE AS BOOLEAN) as is_manual_posting,
    CAST(FALSE AS BOOLEAN) as is_legacy,
    CAST(NULL AS TIMESTAMP) as netsuite_created_timestamp,
    
    -- DWH-specific fields for analysis
    sia.FLG_ACCRUED_TYPE_REGULAR_MRR as dwh_is_regular_mrr,
    sia.M_MRR_EOP_EUR as dwh_mrr_eop_eur,
    sia.M_MRR_EOP_ORIG_CUR as dwh_mrr_eop_orig_cur,
    sia.M_AMOUNT_ORIG_CUR as dwh_amount_orig_cur,
    sia.M_FX_EFFECT_LM as dwh_fx_effect_lm,
    sia.M_FX_EFFECT_LTM as dwh_fx_effect_ltm,
    sia.FLG_NEW_CONTEXT_BILLING_CUSTOMER as dwh_new_customer,
    sia.FLG_CHURN_CONTEXT_BILLING_CUSTOMER as dwh_churn_customer,
    sia.FLG_JOIN_LM as dwh_join_lm,
    sia.FLG_CHURN_LM as dwh_churn_lm,
    sia.FLG_UPSELL_LM as dwh_upsell_lm,
    sia.FLG_DOWNSELL_LM as dwh_downsell_lm,
    prod_seg.PRODUCT_SEGMENT_NAME as dwh_product_segment,
    bus_seg.BUSINESS_SEGMENT_NAME as dwh_business_segment

FROM 
    teamblue.dwh.ssas_f_invoices_accrued_v2 sia
    
    -- Join to f_invoices for business keys
    LEFT JOIN teamblue.dwh.f_invoices fi
        ON sia.PK_INVOICES = fi.PK_INVOICES
    
    -- Date dimensions
    LEFT JOIN teamblue.dwh.d_date d_invoice ON sia.FK_DATE_INVOICE = d_invoice.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_order ON fi.FK_DATE_ORDER = d_order.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_start ON fi.FK_DATE_START = d_start.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_end ON fi.FK_DATE_END = d_end.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_accrued ON sia.FK_DATE_ACCUED = d_accrued.PK_DATE
    
    -- Dimensions
    LEFT JOIN teamblue.dwh.d_customers cust ON sia.FK_CUSTOMERS = cust.PK_CUSTOMERS
    LEFT JOIN teamblue.dwh.d_geography geo ON sia.FK_GEOGRAPHY_CUSTOMER = geo.PK_GEOGRAPHY
    LEFT JOIN teamblue.dwh.d_products prod ON sia.FK_PRODUCTS = prod.PK_PRODUCTS
    LEFT JOIN teamblue.dwh.d_product_segment prod_seg ON sia.FK_PRODUCT_SEGMENT = prod_seg.PK_PRODUCT_SEGMENT
    LEFT JOIN teamblue.dwh.d_budget_brand brand ON prod_seg.BK_BUDGET_BRAND_CODE = brand.BK_BUDGET_BRAND_CODE
    LEFT JOIN teamblue.dwh.d_legal_entity legal_ent ON brand.FK_LEGAL_ENTITY = legal_ent.PK_LEGAL_ENTITY
    LEFT JOIN teamblue.dwh.d_currency curr ON sia.FK_CURRENCY = curr.PK_CURRENCY
    LEFT JOIN teamblue.dwh.d_invoice_source inv_src ON fi.FK_INVOICE_SOURCE = inv_src.PK_INVOICE_SOURCE
    LEFT JOIN teamblue.dwh.d_providers prov ON sia.FK_PROVIDERS = prov.PK_PROVIDERS
    LEFT JOIN teamblue.dwh.d_subscriptions subs ON sia.FK_SUBSCRIPTIONS = subs.PK_SUBSCRIPTIONS
    LEFT JOIN teamblue.dwh.d_business_segment bus_seg ON sia.FK_BUSINESS_SEGMENT = bus_seg.PK_BUSINESS_SEGMENT

WHERE
    CAST(d_accrued.BK_DATE AS DATE) > CURRENT_DATE()
    AND sia.FLG_ACCRUED_TYPE_REGULAR_MRR = TRUE
    AND (fi.FLG_EXCLUDE_IN_REPORTING IS NULL OR fi.FLG_EXCLUDE_IN_REPORTING = FALSE)
;
