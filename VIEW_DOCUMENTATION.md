# Deferred Invoices View Documentation

## Overview
This document describes the `dwh.v_deferred_invoices` view created to represent deferred revenue invoices from the dwh dimensional model.

## Purpose
The view extracts and presents deferred invoices in a format comparable to `teamblue.silver.deferred_revenue` table, facilitating data validation and reconciliation between the dwh dimensional model and the Silver layer.

## Definition of Deferred Invoices
**Deferred invoices** are identified as invoices where:
- The accrual date (`FK_DATE_ACCRUED` from `f_invoices_accrued`) is **in the future** (greater than `CURRENT_DATE()`)

## Source Tables

### Fact Tables
1. **`dwh.f_invoices`** - Main invoice fact table containing invoice transactions
2. **`dwh.f_invoices_accrued`** - Accrual information for invoices including accrual dates and FX effects

### Dimension Tables
The view joins to the following dimension tables to provide descriptive attributes:

| Dimension Table | Purpose | Key Attributes |
|----------------|---------|----------------|
| `d_date` | Date information | Multiple instances for different date types |
| `d_customers` | Customer details | Customer code, name |
| `d_geography` | Geographic regions | Region name |
| `d_legal_entity` | Organizational entities | Entity and sub-entity names |
| `d_products` | Product information | Product name |
| `d_product_segment` | Product categorization | Product segment name |
| `d_budget_brand` | Brand information | Brand name |
| `d_currency` | Currency details | Currency code |
| `d_invoice_type` | Invoice classification | Invoice type name |
| `d_invoice_classification` | Additional invoice classification | Classification name |
| `d_invoice_source` | Invoice source system | Source name |
| `d_providers` | Service providers | Provider name |
| `d_subscriptions` | Subscription types | Subscription type name |
| `d_business_segment` | Business segmentation | Business segment name |

## Column Mapping

The view columns map to `silver.deferred_revenue` as follows:

| View Column | Silver Column | Source | Description |
|-------------|---------------|--------|-------------|
| `region` | `region` | `d_geography.REGION_NAME` | Geographic region |
| `entity` | `entity` | `d_legal_entity.ENTITY_NAME` | Legal entity |
| `sub_entity` | `sub_entity` | `d_legal_entity.SUB_ENTITY_NAME` | Sub-entity if applicable |
| `doc_type` | `doc_type` | `d_invoice_type.INVOICE_TYPE_NAME` | Document/invoice type |
| `invoice_pk` | `invoice_pk` | `f_invoices.PK_INVOICES` | Primary key for invoice |
| `invoice_id` | `invoice_id` | `f_invoices.BK_INVOICE_CODE` | Business key/invoice ID |
| `customer_id` | `customer_id` | `d_customers.CUSTOMER_CODE` | Customer identifier |
| `brand_name` | `brand_name` | `d_budget_brand.BRAND_NAME` | Brand name |
| `currency_code` | `currency_code` | `d_currency.CURRENCY_CODE` | Currency code (EUR, USD, etc.) |
| `transaction_date` | `transaction_date` | `d_date.DATE_VALUE` (invoice) | Transaction date |
| `due_date` | `due_date` | `d_date.DATE_VALUE` (order) | Due date |
| `exchange_rate` | `exchange_rate` | `f_invoices.EXCHANGE_RATE` | Exchange rate |
| `terms` | `terms` | `d_products.PRODUCT_NAME` | Product/contract terms |
| `memo` | `memo` | `f_invoices.BK_INVOICE_LINE_CODE` | Invoice line memo |

## Key Features

### 1. Deferred Revenue Filter
```sql
WHERE d_accrued.DATE_VALUE > CURRENT_DATE()
```
This filter ensures only future-accrued invoices are included.

### 2. Comprehensive Date Tracking
The view includes multiple date dimensions:
- **transaction_date / invoice_date**: When the invoice was issued
- **due_date**: Payment due date
- **start_date / end_date**: Service period dates
- **accrual_date**: When revenue is accrued
- **accrued_start_date / accrued_end_date**: Accrual period
- **renewal_date**: Contract renewal date

### 3. Financial Metrics
- **fx_effect_lm**: Foreign exchange effect for last month
- **fx_effect_ltm**: Foreign exchange effect for last twelve months
- **exchange_rate / exchange_rate_to_eur**: Currency conversion rates

### 4. Classification Flags
- **is_deferred**: Flag indicating if invoice is deferred (1 = yes, 0 = no)
- **is_non_accrued_due_to_zero_mrr**: Flag for non-accrued items due to zero MRR

## Usage Examples

### Query all deferred invoices
```sql
SELECT * 
FROM dwh.v_deferred_invoices
ORDER BY accrual_date;
```

### Compare with Silver layer
```sql
-- Count comparison
SELECT COUNT(*) as dwh_count FROM dwh.v_deferred_invoices;
SELECT COUNT(*) as silver_count FROM teamblue.silver.deferred_revenue;

-- Column-level comparison by brand
SELECT 
    brand_name,
    COUNT(*) as invoice_count,
    SUM(fx_effect_lm) as total_fx_lm
FROM dwh.v_deferred_invoices
GROUP BY brand_name
ORDER BY invoice_count DESC;
```

### Filter by specific brand
```sql
SELECT *
FROM dwh.v_deferred_invoices
WHERE brand_name = 'YourBrandName'
  AND accrual_date BETWEEN '2025-01-01' AND '2025-12-31';
```

### Get deferred revenue by month
```sql
SELECT 
    DATE_TRUNC('month', accrual_date) as accrual_month,
    brand_name,
    COUNT(*) as invoice_count,
    SUM(fx_effect_lm) as total_deferred_revenue
FROM dwh.v_deferred_invoices
GROUP BY DATE_TRUNC('month', accrual_date), brand_name
ORDER BY accrual_month, brand_name;
```

## Important Notes

1. **Join Strategy**: All dimension joins use `LEFT JOIN` to preserve all invoice records even if dimension data is missing.

2. **Brand Dimension**: The brand can come from either the product dimension or directly from the invoice. The view uses an `OR` condition to capture both scenarios:
   ```sql
   ON prod.FK_BUDGET_BRAND = brand.PK_BUDGET_BRAND
      OR fi.FK_BUDGET_BRAND = brand.PK_BUDGET_BRAND
   ```

3. **Performance Considerations**: 
   - The view joins multiple dimension tables, which may impact query performance
   - Consider creating materialized views or indexing for frequently accessed columns
   - The `WHERE d_accrued.DATE_VALUE > CURRENT_DATE()` filter reduces the result set significantly

4. **Data Quality**: 
   - Some invoices may have NULL values in certain dimension fields
   - The `COALESCE` function is used for FX metrics to ensure numeric values
   - Validate data completeness by checking for NULLs in critical fields

5. **Customization**: 
   - Additional filters can be added to the WHERE clause as needed
   - Uncomment optional filters in the SQL file to exclude specific invoice types
   - Modify column selections based on your specific reporting requirements

## Maintenance

### To update the view:
```sql
-- Run the create_v_deferred_invoices.sql script
-- This will replace the existing view with the updated definition
```

### To drop the view:
```sql
DROP VIEW IF EXISTS dwh.v_deferred_invoices;
```

### To check view dependencies:
```sql
-- Find tables/views that reference this view
SHOW DEPENDENCIES dwh.v_deferred_invoices;
```

## Validation Checklist

When validating the view against `silver.deferred_revenue`:

- [ ] Compare row counts between dwh and silver
- [ ] Verify brand distribution matches
- [ ] Check date ranges are consistent
- [ ] Validate currency codes are correctly mapped
- [ ] Ensure customer IDs match between systems
- [ ] Compare aggregated revenue figures (fx_effect_lm/ltm)
- [ ] Verify no unexpected NULL values in key fields
- [ ] Check for duplicate invoice records
- [ ] Validate foreign exchange rates are reasonable

## Contact

For questions about this view or the dwh dimensional model, contact:
- **Data Warehouse Team**: Milos Milenkovic (as mentioned in the conversation)
- **Finance Data Team**: tb-team-data-finance channel

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-18 | AI Assistant | Initial creation based on silver.deferred_revenue structure |
