-- ============================================================================
-- View: teamblue.dwh.v_deferred_invoices
-- ============================================================================
-- Purpose: Create a view representing deferred invoices from the dwh dimensional model
-- Definition: Deferred invoices are those where the accrual date (FK_DATE_ACCRUED) 
--             is in the future compared to the current date
-- Structure: Matches teamblue.silver.deferred_revenue for comparison
-- ============================================================================

CREATE OR REPLACE VIEW teamblue.dwh.v_deferred_invoices AS
SELECT
    -- Geographic and organizational dimensions
    geo.REGION_NAME as region,
    CAST(NULL AS STRING) as entity,  -- FK_LEGAL_ENTITY not found, needs adjustment
    CAST(NULL AS STRING) as sub_entity,  -- FK_LEGAL_ENTITY not found, needs adjustment
    
    -- Invoice classification
    CAST(fi.FK_INVOICE_TYPE AS STRING) as doc_type,  -- d_invoice_type table doesn't exist
    inv_src.INVOICE_SOURCE_NAME as source_file_name,
    
    -- Invoice identifiers
    fi.PK_INVOICES as invoice_pk,
    fi.BK_INVOICE_CODE as invoice_id,
    
    -- Internal ID for traceability
    fi.PK_INVOICES as internal_id,
    
    -- Customer information
    cust.BK_CUSTOMER_CODE as customer_id,
    cust.CUSTOMER_NAME as billing_system_customer_id,
    
    -- Brand information
    brand.BRAND_NAME as brand_name,
    
    -- Currency and exchange rates
    curr.CURRENCY_CODE as currency_code,
    fi.EXCHANGE_RATE as exchange_rate,
    fi.EXCHANGE_RATE as exchange_rate_to_eur,
    
    -- Date dimensions
    d_invoice.BK_DATE as transaction_date,
    d_invoice.BK_DATE as invoice_date,
    d_order.BK_DATE as due_date,
    d_start.BK_DATE as start_date,
    d_end.BK_DATE as end_date,
    d_accrued.BK_DATE as accrual_date,
    d_accrued_start.BK_DATE as accrued_start_date,
    d_accrued_end.BK_DATE as accrued_end_date,
    d_renewal.BK_DATE as renewal_date,
    
    -- Product and subscription information
    prod.PRODUCT_NAME as terms,
    prod_seg.PRODUCT_SEGMENT_NAME as product_segment,
    subs.SUBSCRIPTION_TYPE_NAME as subscription_type,
    
    -- Provider information  
    prov.PROVIDER_NAME as provider_name,
    
    -- Invoice amounts and metrics (from f_invoices_accrued if available)
    COALESCE(fia.FX_EFFECT_LM, 0) as fx_effect_lm,
    COALESCE(fia.FX_EFFECT_LTM, 0) as fx_effect_ltm,
    fia.ACCRUED_TYPE as accrued_type,
    
    -- Additional metadata
    fi.BK_INVOICE_LINE_CODE as memo,
    CAST(fi.FK_INVOICE_CLASSIFICATION AS STRING) as invoice_classification,  -- d_invoice_classification table doesn't exist
    
    -- Business segment
    bus_seg.BUSINESS_SEGMENT_NAME as business_segment,
    
    -- Workato metadata
    CURRENT_TIMESTAMP() as workato_log_timestamp,
    
    -- Flags for deferred revenue logic
    CASE 
        WHEN d_accrued.BK_DATE > CURRENT_DATE() THEN 1 
        ELSE 0 
    END as is_deferred,
    
    CASE 
        WHEN fia.FLG_IS_NON_ACCRUED_DUE_TO_ZERO_MRR = 1 THEN 1
        ELSE 0
    END as is_non_accrued_due_to_zero_mrr

FROM 
    teamblue.dwh.f_invoices fi
    
    -- Join with invoices_accrued fact table
    LEFT JOIN teamblue.dwh.f_invoices_accrued fia
        ON fi.PK_INVOICES = fia.PK_INVOICES
    
    -- Date dimensions
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
    LEFT JOIN teamblue.dwh.d_date d_accrued_start
        ON fi.FK_DATE_ACCRUED_START = d_accrued_start.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_accrued_end
        ON fi.FK_DATE_ACCRUED_END = d_accrued_end.PK_DATE
    LEFT JOIN teamblue.dwh.d_date d_renewal
        ON fi.FK_DATE_UP_FOR_RENEWAL = d_renewal.PK_DATE
    
    -- Customer dimensions
    LEFT JOIN teamblue.dwh.d_customers cust
        ON fi.FK_CUSTOMERS = cust.PK_CUSTOMERS
    
    -- Geography dimension (directly from fact table)
    LEFT JOIN teamblue.dwh.d_geography geo
        ON fi.FK_GEOGRAPHY_CUSTOMER = geo.PK_GEOGRAPHY
    
    -- Organization/Legal Entity dimension (if exists in your schema)
    -- Note: FK_LEGAL_ENTITY was not found in d_customers, you may need to adjust this join
    -- LEFT JOIN teamblue.dwh.d_legal_entity org
    --     ON cust.FK_LEGAL_ENTITY = org.PK_LEGAL_ENTITY
    
    -- Product dimensions
    LEFT JOIN teamblue.dwh.d_products prod
        ON fi.FK_PRODUCTS = prod.PK_PRODUCTS
    LEFT JOIN teamblue.dwh.d_product_segment prod_seg
        ON fi.FK_PRODUCT_SEGMENT = prod_seg.PK_PRODUCT_SEGMENT
    
    -- Brand dimension (through product segment)
    LEFT JOIN teamblue.dwh.d_budget_brand brand
        ON prod_seg.BK_BUDGET_BRAND_CODE = brand.BK_BUDGET_BRAND_CODE
    
    -- Currency dimension
    LEFT JOIN teamblue.dwh.d_currency curr
        ON fi.FK_CURRENCY = curr.PK_CURRENCY
    
    -- Invoice source (d_invoice_type and d_invoice_classification tables don't exist)
    LEFT JOIN teamblue.dwh.d_invoice_source inv_src
        ON fi.FK_INVOICE_SOURCE = inv_src.PK_INVOICE_SOURCE
    
    -- Provider dimension
    LEFT JOIN teamblue.dwh.d_providers prov
        ON fi.FK_PROVIDERS = prov.PK_PROVIDERS
    
    -- Subscription dimension
    LEFT JOIN teamblue.dwh.d_subscriptions subs
        ON fi.FK_SUBSCRIPTIONS = subs.PK_SUBSCRIPTIONS
    
    -- Business segment dimension (from fact table)
    LEFT JOIN teamblue.dwh.d_business_segment bus_seg
        ON fi.FK_BUSINESS_SEGMENT = bus_seg.PK_BUSINESS_SEGMENT

WHERE
    -- Filter for deferred invoices only (accrual date in the future)
    d_accrued.BK_DATE > CURRENT_DATE()
    
    -- Optionally, you may want to include additional filters:
    -- AND inv_type.INVOICE_TYPE_NAME NOT IN ('Credit Note', 'Refund')
    -- AND fia.ACCRUED_TYPE = 'DEFERRED'  -- if such a type exists
;

-- ============================================================================
-- Usage Notes:
-- ============================================================================
-- 1. This view identifies deferred invoices based on FK_DATE_ACCRUED > CURRENT_DATE()
-- 2. The structure matches silver.deferred_revenue for comparison purposes
-- 3. All dimension joins are LEFT JOINs to preserve all invoice records
-- 4. Additional filters can be added in the WHERE clause as needed
-- 5. The view assumes standard naming conventions for dwh dimension tables
-- ============================================================================
