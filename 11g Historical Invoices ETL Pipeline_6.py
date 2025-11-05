# Databricks notebook source
# ============================================================================
# 11G Historical Invoices ETL Pipeline
# ============================================================================
# Loads and transforms invoice and invoice line data from CSV files
# Target: teamblue.findata_sandbox staging tables

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Configuration
INVOICES_PATH = "/Volumes/teamblue/default/teamblue-client/sandbox_files/belgium/COM_20251023_144040_invoices.csv"
INVOICE_LINES_PATH = "/Volumes/teamblue/default/teamblue-client/sandbox_files/belgium/COM_20251023_144040_invoicelines.csv"
TARGET_SCHEMA = "teamblue.findata_sandbox"
PARTITIONS = 200
MAX_RECORDS_PER_FILE = 100000

# COMMAND ----------

# ============================================================================
# Helper Functions
# ============================================================================

def load_csv(path, description):
    """Load CSV with standard options and clean encoding issues"""
    print(f"Loading {description}...")
    
    df = (spark.read
        .option("header", "true")
        .option("delimiter", ";")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiline", "true")
        .option("encoding", "UTF-8")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .option("mode", "PERMISSIVE")
        .csv(path))
    
    # Fix BOM encoding issue in first column
    first_col = df.columns[0]
    if first_col.strip() != "InvoiceId" and "InvoiceId" not in df.columns:
        df = df.withColumnRenamed(first_col, "InvoiceId")
    elif first_col.strip() != "InvoiceId":
        df = df.drop(first_col)
    
    # Clean data: remove quotes and newlines from all columns
    for col_name in df.columns:
        df = df.withColumn(
            col_name,
            trim(regexp_replace(regexp_replace(col(col_name), '"', ''), '\\n', ''))
        )
    
    print(f"  Loaded {df.count():,} records with {len(df.columns)} columns")
    return df


def parse_date(date_col):
    """Parse European date format (d/MM/yyyy H:mm:ss) to date"""
    return to_date(regexp_replace(col(date_col), " \\d+:\\d+:\\d+$", ""), "d/MM/yyyy")


def parse_decimal(decimal_col):
    """Convert European decimal format (comma) to standard format (dot)"""
    return regexp_replace(col(decimal_col), ",", ".")

# COMMAND ----------

# ============================================================================
# Load and Transform Invoices
# ============================================================================

print("="*80)
print("PROCESSING INVOICES")
print("="*80)

# Load raw data
invoices_raw = load_csv(INVOICES_PATH, "invoices")

# Transform
invoices_transformed = (invoices_raw
    .select(
        col("InvoiceId").cast(LongType()),
        col("InvoiceNumber").cast(LongType()),
        col("ProviderId").cast(LongType()),
        col("CustomerKey").cast(LongType()),
        parse_date("InvoiceDate").alias("InvoiceDate"),
        parse_decimal("TaxPercentage").cast(LongType()).alias("TaxPercentage"),
        col("Currency"),
        col("ExchangeRate").cast(LongType()),
        col("BillingAddressee"),
        col("BillingStreet"),
        col("BillingStreetNumber"),
        col("BillingPostalCode"),
        col("BillingCity"),
        col("BillingCountryCode"),
        col("IsCreditNote").cast(LongType()),
        col("OGM")
    )
    .repartition(PARTITIONS)
)

# Data quality checks
null_dates = invoices_transformed.filter(col("InvoiceDate").isNull()).count()
if null_dates > 0:
    print(f"  WARNING: {null_dates:,} invoices with null dates")

# Write to table
print(f"  Writing to {TARGET_SCHEMA}.stg_11g_hist_invoices...")
(invoices_transformed
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("maxRecordsPerFile", MAX_RECORDS_PER_FILE)
    .saveAsTable(f"{TARGET_SCHEMA}.stg_11g_hist_invoices"))

# Verify
invoices_count = spark.table(f"{TARGET_SCHEMA}.stg_11g_hist_invoices").count()
print(f"  ✓ Loaded {invoices_count:,} invoices")

# Display sample
display(spark.table(f"{TARGET_SCHEMA}.stg_11g_hist_invoices").limit(5))

# COMMAND ----------

# ============================================================================
# Load and Transform Invoice Lines
# ============================================================================

print("="*80)
print("PROCESSING INVOICE LINES")
print("="*80)

# Load raw data
invoice_lines_raw = load_csv(INVOICE_LINES_PATH, "invoice lines")

# Check if InvoiceId exists in CSV or needs to be joined
has_invoice_id = "InvoiceId" in invoice_lines_raw.columns

if has_invoice_id:
    print("  Using InvoiceId from CSV")
    invoice_lines_transformed = (invoice_lines_raw
        .select(
            col("InvoiceId").cast(LongType()),
            col("InvoiceLineId").cast(LongType()),
            parse_decimal("UnitPrice").cast(DoubleType()).alias("UnitPrice"),
            parse_decimal("Quantity").cast(DoubleType()).alias("Quantity"),
            parse_decimal("TaxPercentage").cast(DoubleType()).alias("TaxPercentage"),
            parse_decimal("TaxTotal").cast(DoubleType()).alias("TaxTotal"),
            parse_decimal("TotalTaxExcl").cast(DoubleType()).alias("TotalTaxExcl"),
            parse_decimal("TotalTaxIncl").cast(DoubleType()).alias("TotalTaxIncl"),
            parse_date("StartDate").alias("StartDate"),
            parse_date("EndDate").alias("EndDate"),
            col("Description"),
            col("ProductCode"),
            col("ProductGroupCode")
        )
    )
else:
    print("  Joining with invoices table to get InvoiceId")
    # Join with invoices to get InvoiceId via InvoiceNumber
    invoices_lookup = spark.table(f"{TARGET_SCHEMA}.stg_11g_hist_invoices").select("InvoiceId", "InvoiceNumber")
    
    invoice_lines_transformed = (invoice_lines_raw
        .select(
            col("InvoiceNumber").cast(LongType()),
            col("InvoiceLineId").cast(LongType()),
            parse_decimal("UnitPrice").cast(DoubleType()).alias("UnitPrice"),
            parse_decimal("Quantity").cast(DoubleType()).alias("Quantity"),
            parse_decimal("TaxPercentage").cast(DoubleType()).alias("TaxPercentage"),
            parse_decimal("TaxTotal").cast(DoubleType()).alias("TaxTotal"),
            parse_decimal("TotalTaxExcl").cast(DoubleType()).alias("TotalTaxExcl"),
            parse_decimal("TotalTaxIncl").cast(DoubleType()).alias("TotalTaxIncl"),
            parse_date("StartDate").alias("StartDate"),
            parse_date("EndDate").alias("EndDate"),
            col("Description"),
            col("ProductCode"),
            col("ProductGroupCode")
        )
        .join(invoices_lookup, "InvoiceNumber", "left")
        .drop("InvoiceNumber")
    )

invoice_lines_transformed = invoice_lines_transformed.repartition(PARTITIONS)

# Data quality checks
null_invoice_ids = invoice_lines_transformed.filter(col("InvoiceId").isNull()).count()
if null_invoice_ids > 0:
    print(f"  WARNING: {null_invoice_ids:,} line items with null InvoiceId")

# Write to table
print(f"  Writing to {TARGET_SCHEMA}.stg_11g_hist_invoice_items...")
(invoice_lines_transformed
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("maxRecordsPerFile", MAX_RECORDS_PER_FILE)
    .saveAsTable(f"{TARGET_SCHEMA}.stg_11g_hist_invoice_items"))

# Verify
lines_count = spark.table(f"{TARGET_SCHEMA}.stg_11g_hist_invoice_items").count()
print(f"  ✓ Loaded {lines_count:,} invoice line items")

# Display sample
display(spark.table(f"{TARGET_SCHEMA}.stg_11g_hist_invoice_items").limit(5))

# COMMAND ----------

# ============================================================================
# Data Quality Validation
# ============================================================================

print("="*80)
print("DATA QUALITY VALIDATION")
print("="*80)

invoices_table = spark.table(f"{TARGET_SCHEMA}.stg_11g_hist_invoices")
line_items_table = spark.table(f"{TARGET_SCHEMA}.stg_11g_hist_invoice_items")

# Summary statistics
invoice_stats = invoices_table.agg(
    count("*").alias("total_invoices"),
    min("InvoiceDate").alias("min_date"),
    max("InvoiceDate").alias("max_date"),
    countDistinct("CustomerKey").alias("unique_customers"),
    countDistinct("ProviderId").alias("unique_providers")
).first()

line_stats = line_items_table.agg(
    count("*").alias("total_lines"),
    countDistinct("InvoiceId").alias("unique_invoices"),
    sum("Quantity").alias("total_quantity")
).first()

print(f"\nInvoices Summary:")
print(f"  Total records    : {invoice_stats['total_invoices']:,}")
print(f"  Date range       : {invoice_stats['min_date']} to {invoice_stats['max_date']}")
print(f"  Unique customers : {invoice_stats['unique_customers']:,}")
print(f"  Unique providers : {invoice_stats['unique_providers']:,}")

print(f"\nLine Items Summary:")
print(f"  Total records     : {line_stats['total_lines']:,}")
print(f"  Unique invoices   : {line_stats['unique_invoices']:,}")
print(f"  Total quantity    : {line_stats['total_quantity']:,}")

# Referential integrity check
orphaned_lines = line_items_table.join(invoices_table, "InvoiceId", "left_anti").count()
invoices_without_lines = invoices_table.join(line_items_table, "InvoiceId", "left_anti").count()

print(f"\nReferential Integrity:")
print(f"  Orphaned line items         : {orphaned_lines:,}")
print(f"  Invoices without line items : {invoices_without_lines:,}")

if orphaned_lines == 0 and invoices_without_lines == 0:
    print(f"  Status: ✓ PASS")
elif orphaned_lines == 0:
    print(f"  Status: ⚠ WARNING (some invoices have no lines)")
else:
    print(f"  Status: ✗ FAIL (orphaned line items found)")

# Records by year
print("\nRecords by Year:")
year_distribution = (invoices_table
    .filter(col("InvoiceDate").isNotNull())
    .withColumn("year", year("InvoiceDate"))
    .groupBy("year")
    .agg(
        count("*").alias("invoice_count"),
        countDistinct("CustomerKey").alias("unique_customers")
    )
    .orderBy("year"))

display(year_distribution)

# Top countries
print("\nTop 10 Countries by Invoice Count:")
country_distribution = (invoices_table
    .filter(col("BillingCountryCode").isNotNull())
    .groupBy("BillingCountryCode")
    .agg(count("*").alias("invoice_count"))
    .orderBy(col("invoice_count").desc())
    .limit(10))

display(country_distribution)

print("\n" + "="*80)
print("ETL PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)

# COMMAND ----------
