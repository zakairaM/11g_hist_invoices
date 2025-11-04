# Databricks notebook source
spark.sql("DESCRIBE teamblue.findata_sandbox.stg_11g_hist_invoices").show(truncate=False)
spark.sql("DESCRIBE teamblue.findata_sandbox.stg_11g_hist_invoice_items").show(truncate=False)


# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from IPython.display import display

print("="*80)
print("LOADING CSV FILE")
print("="*80)

try:
    # Read CSV 
    invoices_raw_sdf = (spark
        .read
        .option("header", "true")
        .option("delimiter", ";")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiline", "true")
        .option("encoding", "UTF-8")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .option("mode", "PERMISSIVE")
        .csv("/Volumes/teamblue/default/teamblue-client/sandbox_files/belgium/COM_20251023_144040_invoices.csv"))
    
    # Get the first column name (which has encoding issues)
    first_col = invoices_raw_sdf.columns[0]
    
    # Check if first column is already "InvoiceId" (ignoring BOM/encoding)
    # If it looks different due to encoding, we need to handle it carefully
    if first_col.strip() != "InvoiceId":
        # Check if "InvoiceId" already exists in other columns
        if "InvoiceId" in invoices_raw_sdf.columns:
            # Drop the problematic first column and keep the existing InvoiceId
            invoices_raw_sdf = invoices_raw_sdf.drop(first_col)
            print(f"[SUCCESS] Dropped problematic first column '{first_col}', using existing 'InvoiceId' column")
        else:
            # Rename the first column
            invoices_raw_sdf = invoices_raw_sdf.withColumnRenamed(first_col, "InvoiceId")
            print(f"[SUCCESS] Renamed problematic first column to 'InvoiceId'")
    else:
        print(f"[INFO] First column is already 'InvoiceId', no rename needed")
    
    # Now clean the DATA in each column (remove quotes and newlines from values)
    for col_name in invoices_raw_sdf.columns:
        invoices_raw_sdf = invoices_raw_sdf.withColumn(
            col_name,
            trim(regexp_replace(regexp_replace(invoices_raw_sdf[col_name], '"', ''), '\\n', ''))
        )
    
    # Cache the cleaned dataframe
    invoices_raw_sdf.cache()
    
    record_count = invoices_raw_sdf.count()
    column_count = len(invoices_raw_sdf.columns)
    
    print(f"\n[SUCCESS] CSV loaded and cleaned successfully")
    print(f"  Records: {record_count:,}")
    print(f"  Columns: {column_count}")
    
    # Display sample data
    print("\n" + "="*80)
    print("SAMPLE DATA (First 5 rows)")
    print("="*80)
    display(invoices_raw_sdf.limit(5))
    
    # Show schema
    print("\n" + "="*80)
    print("SCHEMA SUMMARY")
    print("="*80)
    for field in invoices_raw_sdf.schema.fields:
        print(f"  {field.name:30s} : {field.dataType}")
    
    # Data quality checks
    print("\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)
    
    # Check for null or empty values in key columns
    key_columns = ["InvoiceId", "InvoiceNumber", "CustomerKey", "InvoiceDate"]
    for col_name in key_columns:
        null_count = invoices_raw_sdf.filter(
            col(col_name).isNull() | (trim(col(col_name)) == "")
        ).count()
        print(f"  {col_name:30s} : {null_count:,} nulls/empty")
    
    # Show sample of cleaned data
    print(f"\n  Sample cleaned values:")
    invoices_raw_sdf.select("InvoiceId", "InvoiceNumber", "InvoiceDate", "TaxPercentage", "Currency").show(10, truncate=False)
    
    # Show date range
    print(f"\n  Date Range:")
    invoices_raw_sdf.agg(
        min("InvoiceDate").alias("min_date"),
        max("InvoiceDate").alias("max_date")
    ).show(truncate=False)
    
    print("\n" + "="*80)
    print("[SUCCESS] Data loaded and cached successfully")
    print("="*80)
    
except Exception as e:
    print(f"\n[ERROR] Error loading CSV: {e}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

print("="*80)
print("TRANSFORMING INVOICE DATA")
print("="*80)

# First, let's examine the actual date values closely
print("\n[DEBUG] Examining date format in detail:")
date_samples = invoices_raw_sdf.select("InvoiceDate").limit(10).collect()
for row in date_samples[:5]:
    date_val = row["InvoiceDate"]
    print(f"  Raw value: '{date_val}' | Length: {len(date_val) if date_val else 0} | Repr: {repr(date_val)}")

# Try parsing with different methods
print("\n[DEBUG] Testing different parsing methods:")
test_df = invoices_raw_sdf.limit(5).select(
    col("InvoiceDate").alias("original"),
    to_date(col("InvoiceDate"), "d/MM/yyyy HH:mm:ss").alias("method1"),
    to_date(col("InvoiceDate"), "d/MM/yyyy H:mm:ss").alias("method2"),
    to_date(to_timestamp(col("InvoiceDate"), "d/MM/yyyy H:mm:ss")).alias("method3"),
    to_date(regexp_replace(col("InvoiceDate"), " 0:00:00$", ""), "d/MM/yyyy").alias("method4"),
)
test_df.show(truncate=False)

# Use the method that works
invoices_final_sdf = (invoices_raw_sdf
    .select(
        # Cast numeric fields
        col("InvoiceId").cast(LongType()).alias("InvoiceId"),
        col("InvoiceNumber").cast(LongType()).alias("InvoiceNumber"),
        col("ProviderId").cast(LongType()).alias("ProviderId"),
        col("CustomerKey").cast(LongType()).alias("CustomerKey"),
        
        # Parse date - remove the time part first, then parse
        to_date(
            regexp_replace(col("InvoiceDate"), " \\d+:\\d+:\\d+$", ""),
            "d/MM/yyyy"
        ).alias("InvoiceDate"),
        
        # Cast TaxPercentage - replace comma with dot for decimal
        regexp_replace(col("TaxPercentage"), ",", ".").cast(LongType()).alias("TaxPercentage"),
        
        # Clean string fields
        col("Currency").alias("Currency"),
        col("ExchangeRate").cast(LongType()).alias("ExchangeRate"),
        col("BillingAddressee").alias("BillingAddressee"),
        col("BillingStreet").alias("BillingStreet"),
        col("BillingStreetNumber").alias("BillingStreetNumber"),
        col("BillingPostalCode").alias("BillingPostalCode"),
        col("BillingCity").alias("BillingCity"),
        col("BillingCountryCode").alias("BillingCountryCode"),
        col("IsCreditNote").cast(LongType()).alias("IsCreditNote"),
        col("OGM").alias("OGM"),
        
        # Add df_source field
        lit("COM_20251023_144040").alias("df_source")
    )
)

# Verify date parsing worked
print("\n[DEBUG] Checking parsed dates:")
invoices_final_sdf.select("InvoiceDate").show(10, truncate=False)

null_dates = invoices_final_sdf.filter(col("InvoiceDate").isNull()).count()
print(f"\n[INFO] Null dates after parsing: {null_dates:,}")

# Repartition for optimal write performance
invoices_final_sdf = invoices_final_sdf.repartition(200)

print("\n[SUCCESS] Transformed invoice data")

# Display transformed schema
print("\n" + "="*80)
print("TRANSFORMED SCHEMA")
print("="*80)
for field in invoices_final_sdf.schema.fields:
    print(f"  {field.name:30s} : {field.dataType}")

# Display sample transformed data
print("\n" + "="*80)
print("SAMPLE TRANSFORMED DATA (First 5 rows)")
print("="*80)
display(invoices_final_sdf.limit(5))

# Load invoices to target table
print("\n" + "="*80)
print("LOADING INVOICE DATA TO TARGET TABLE")
print("="*80)

try:
    print("\n[INFO] Writing data to table...")
    print("  Target: teamblue.findata_sandbox.stg_11g_hist_invoices")
    print("  Mode: overwrite")
    print("  Partitions: 200")
    print("  Max records per file: 100,000")
    
    (invoices_final_sdf
        .write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("maxRecordsPerFile", 100000)
        .saveAsTable("teamblue.findata_sandbox.stg_11g_hist_invoices"))
    
    print("\n[SUCCESS] Data loaded successfully")
    
    # Verify count
    invoices_loaded_count = spark.table("teamblue.findata_sandbox.stg_11g_hist_invoices").count()
    print(f"\n  Records loaded: {invoices_loaded_count:,}")
    
    # Verify data quality
    print("\n" + "="*80)
    print("DATA QUALITY VALIDATION")
    print("="*80)
    
    loaded_table = spark.table("teamblue.findata_sandbox.stg_11g_hist_invoices")
    
    # Check for null values in key columns
    key_columns = ["InvoiceId", "InvoiceNumber", "CustomerKey", "InvoiceDate"]
    for col_name in key_columns:
        null_count = loaded_table.filter(col(col_name).isNull()).count()
        print(f"  {col_name:30s} : {null_count:,} nulls")
    
    # Show date range
    date_stats = loaded_table.agg(
        min("InvoiceDate").alias("min_date"),
        max("InvoiceDate").alias("max_date"),
        countDistinct("InvoiceId").alias("distinct_invoices")
    ).first()
    
    print(f"\n  Date Range:")
    print(f"    Earliest: {date_stats['min_date']}")
    print(f"    Latest  : {date_stats['max_date']}")
    print(f"\n  Distinct Invoices: {date_stats['distinct_invoices']:,}")
    
    # Display sample from loaded table
    print("\n" + "="*80)
    print("SAMPLE FROM LOADED TABLE (First 5 rows)")
    print("="*80)
    display(spark.table("teamblue.findata_sandbox.stg_11g_hist_invoices").limit(5))
    
    # Show row count by year (only non-null dates)
    print("\n" + "="*80)
    print("RECORDS BY YEAR")
    print("="*80)
    year_distribution = (loaded_table
        .filter(col("InvoiceDate").isNotNull())
        .withColumn("year", year("InvoiceDate"))
        .groupBy("year")
        .count()
        .orderBy("year"))
    
    display(year_distribution)
    
    print("\n" + "="*80)
    print("[SUCCESS] Invoice data loaded and validated")
    print("="*80)
    
except Exception as e:
    print(f"\n[ERROR] Error loading invoice data: {e}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

print("="*80)
print("LOADING INVOICE LINES CSV FILE")
print("="*80)

try:
    # Read Invoice Lines CSV 
    invoice_lines_raw_sdf = (spark
        .read
        .option("header", "true")
        .option("delimiter", ";")
        .option("quote", '"')
        .option("escape", '"')
        .option("multiline", "true")
        .option("encoding", "UTF-8")
        .option("ignoreLeadingWhiteSpace", "true")
        .option("ignoreTrailingWhiteSpace", "true")
        .option("mode", "PERMISSIVE")
        .csv("/Volumes/teamblue/default/teamblue-client/sandbox_files/belgium/COM_20251023_144040_invoicelines.csv"))
    
    # Get the first column name (which may have encoding issues)
    first_col = invoice_lines_raw_sdf.columns[0]
    
    # Check if first column is already "InvoiceLineId" (ignoring BOM/encoding)
    # If it looks different due to encoding, we need to handle it carefully
    if first_col.strip() != "InvoiceLineId":
        # Check if "InvoiceLineId" already exists in other columns
        if "InvoiceLineId" in invoice_lines_raw_sdf.columns:
            # Drop the problematic first column and keep the existing InvoiceLineId
            invoice_lines_raw_sdf = invoice_lines_raw_sdf.drop(first_col)
            print(f"[SUCCESS] Dropped problematic first column '{first_col}', using existing 'InvoiceLineId' column")
        else:
            # Rename the first column
            invoice_lines_raw_sdf = invoice_lines_raw_sdf.withColumnRenamed(first_col, "InvoiceLineId")
            print(f"[SUCCESS] Renamed problematic first column to 'InvoiceLineId'")
    else:
        print(f"[INFO] First column is already 'InvoiceLineId', no rename needed")
    
    # Clean the DATA in each column (remove quotes and newlines from values)
    for col_name in invoice_lines_raw_sdf.columns:
        invoice_lines_raw_sdf = invoice_lines_raw_sdf.withColumn(
            col_name,
            trim(regexp_replace(regexp_replace(invoice_lines_raw_sdf[col_name], '"', ''), '\\n', ''))
        )
    
    # Cache the cleaned dataframe
    invoice_lines_raw_sdf.cache()
    
    record_count = invoice_lines_raw_sdf.count()
    column_count = len(invoice_lines_raw_sdf.columns)
    
    print(f"\n[SUCCESS] Invoice Lines CSV loaded and cleaned successfully")
    print(f"  Records: {record_count:,}")
    print(f"  Columns: {column_count}")
    
    # Display sample data
    print("\n" + "="*80)
    print("SAMPLE DATA (First 5 rows)")
    print("="*80)
    display(invoice_lines_raw_sdf.limit(5))
    
    # Show schema
    print("\n" + "="*80)
    print("SCHEMA SUMMARY")
    print("="*80)
    print(f"\n[DEBUG] All columns in invoice lines CSV:")
    print(f"  {invoice_lines_raw_sdf.columns}")
    print("\n")
    for field in invoice_lines_raw_sdf.schema.fields:
        print(f"  {field.name:30s} : {field.dataType}")
    
    # Data quality checks
    print("\n" + "="*80)
    print("DATA QUALITY CHECKS")
    print("="*80)
    
    # Check for null or empty values in key columns
    # Note: InvoiceId might not be in raw data - will be added via join
    key_columns = ["InvoiceLineId"]
    if "InvoiceId" in invoice_lines_raw_sdf.columns:
        key_columns.append("InvoiceId")
    if "InvoiceNumber" in invoice_lines_raw_sdf.columns:
        key_columns.append("InvoiceNumber")
        
    for col_name in key_columns:
        null_count = invoice_lines_raw_sdf.filter(
            col(col_name).isNull() | (trim(col(col_name)) == "")
        ).count()
        print(f"  {col_name:30s} : {null_count:,} nulls/empty")
    
    # Show sample of cleaned data
    print(f"\n  Sample cleaned values:")
    # Build sample columns list dynamically based on what exists
    sample_cols = ["InvoiceLineId"]
    if "InvoiceId" in invoice_lines_raw_sdf.columns:
        sample_cols.append("InvoiceId")
    if "InvoiceNumber" in invoice_lines_raw_sdf.columns:
        sample_cols.append("InvoiceNumber")
    for col in ["Description", "UnitPrice", "Quantity"]:
        if col in invoice_lines_raw_sdf.columns:
            sample_cols.append(col)
    
    if sample_cols:
        invoice_lines_raw_sdf.select(*sample_cols).show(10, truncate=False)
    
    print("\n" + "="*80)
    print("[SUCCESS] Invoice Lines data loaded and cached successfully")
    print("="*80)
    
except Exception as e:
    print(f"\n[ERROR] Error loading Invoice Lines CSV: {e}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

print("\n" + "="*80)
print("TRANSFORMING INVOICE LINE ITEMS DATA")
print("="*80)

# First, check all columns in invoice lines for potential join keys
print("\n[INFO] Detecting join key in invoice lines...")
print(f"  Available columns: {invoice_lines_raw_sdf.columns}")

# Look for any column that might be a join key (check all column names)
join_key = None
for col_name in invoice_lines_raw_sdf.columns:
    cleaned_col = col_name.strip()
    if "InvoiceId" in cleaned_col or "invoiceid" in cleaned_col.lower():
        join_key = col_name
        join_type = "InvoiceId"
        break
    elif "InvoiceNumber" in cleaned_col or "invoicenumber" in cleaned_col.lower():
        join_key = col_name
        join_type = "InvoiceNumber"
        break

if join_key:
    print(f"  Found join key: '{join_key}' (will use as {join_type})")
else:
    print(f"  No join key found - will load from existing invoices dataframe")

# Transform invoice lines with standard fields
print("\n[INFO] Transforming invoice line items...")
line_items_transformed_sdf = (invoice_lines_raw_sdf
    .select(
        col("InvoiceLineId").cast(LongType()).alias("InvoiceLineId"),
        
        # Pricing fields - replace comma with dot for European decimal format
        regexp_replace(col("UnitPrice"), ",", ".").cast(DoubleType()).alias("UnitPrice"),
        regexp_replace(col("Quantity"), ",", ".").cast(DoubleType()).alias("Quantity"),
        
        # Tax fields
        regexp_replace(col("TaxPercentage"), ",", ".").cast(DoubleType()).alias("TaxPercentage"),
        regexp_replace(col("TaxTotal"), ",", ".").cast(DoubleType()).alias("TaxTotal"),
        
        # Total fields
        regexp_replace(col("TotalTaxExcl"), ",", ".").cast(DoubleType()).alias("TotalTaxExcl"),
        regexp_replace(col("TotalTaxIncl"), ",", ".").cast(DoubleType()).alias("TotalTaxIncl"),
        
        # Date fields
        to_date(regexp_replace(col("StartDate"), " \\d+:\\d+:\\d+$", ""), "d/MM/yyyy").alias("StartDate"),
        to_date(regexp_replace(col("EndDate"), " \\d+:\\d+:\\d+$", ""), "d/MM/yyyy").alias("EndDate"),
        
        # Description and product fields
        col("Description").alias("Description"),
        col("ProductCode").alias("ProductCode"),
        col("ProductGroupCode").alias("ProductGroupCode")
    )
)

# Add InvoiceId by joining with invoices table if needed
if join_key and join_type == "InvoiceId":
    # InvoiceId already exists in source - just add it
    print("\n[INFO] InvoiceId found in source - adding to transformation")
    line_items_final_sdf = (invoice_lines_raw_sdf
        .select(
            col(join_key).cast(LongType()).alias("InvoiceId"),
            col("InvoiceLineId").cast(LongType()).alias("InvoiceLineId"),
            regexp_replace(col("UnitPrice"), ",", ".").cast(DoubleType()).alias("UnitPrice"),
            regexp_replace(col("Quantity"), ",", ".").cast(DoubleType()).alias("Quantity"),
            regexp_replace(col("TaxPercentage"), ",", ".").cast(DoubleType()).alias("TaxPercentage"),
            regexp_replace(col("TaxTotal"), ",", ".").cast(DoubleType()).alias("TaxTotal"),
            regexp_replace(col("TotalTaxExcl"), ",", ".").cast(DoubleType()).alias("TotalTaxExcl"),
            regexp_replace(col("TotalTaxIncl"), ",", ".").cast(DoubleType()).alias("TotalTaxIncl"),
            to_date(regexp_replace(col("StartDate"), " \\d+:\\d+:\\d+$", ""), "d/MM/yyyy").alias("StartDate"),
            to_date(regexp_replace(col("EndDate"), " \\d+:\\d+:\\d+$", ""), "d/MM/yyyy").alias("EndDate"),
            col("Description").alias("Description"),
            col("ProductCode").alias("ProductCode"),
            col("ProductGroupCode").alias("ProductGroupCode")
        )
    )
elif join_key and join_type == "InvoiceNumber":
    # Use InvoiceNumber to join with invoices table
    print("\n[INFO] Joining with invoices table on InvoiceNumber...")
    line_items_with_number = (invoice_lines_raw_sdf
        .select(
            col(join_key).cast(LongType()).alias("InvoiceNumber"),
            col("InvoiceLineId").cast(LongType()).alias("InvoiceLineId"),
            regexp_replace(col("UnitPrice"), ",", ".").cast(DoubleType()).alias("UnitPrice"),
            regexp_replace(col("Quantity"), ",", ".").cast(DoubleType()).alias("Quantity"),
            regexp_replace(col("TaxPercentage"), ",", ".").cast(DoubleType()).alias("TaxPercentage"),
            regexp_replace(col("TaxTotal"), ",", ".").cast(DoubleType()).alias("TaxTotal"),
            regexp_replace(col("TotalTaxExcl"), ",", ".").cast(DoubleType()).alias("TotalTaxExcl"),
            regexp_replace(col("TotalTaxIncl"), ",", ".").cast(DoubleType()).alias("TotalTaxIncl"),
            to_date(regexp_replace(col("StartDate"), " \\d+:\\d+:\\d+$", ""), "d/MM/yyyy").alias("StartDate"),
            to_date(regexp_replace(col("EndDate"), " \\d+:\\d+:\\d+$", ""), "d/MM/yyyy").alias("EndDate"),
            col("Description").alias("Description"),
            col("ProductCode").alias("ProductCode"),
            col("ProductGroupCode").alias("ProductGroupCode")
        )
    )
    
    # Get InvoiceId from invoices table
    invoices_for_join = invoices_final_sdf.select("InvoiceId", "InvoiceNumber")
    
    line_items_final_sdf = (line_items_with_number
        .join(invoices_for_join, "InvoiceNumber", "left")
        .select(
            col("InvoiceId"),
            col("InvoiceLineId"),
            col("UnitPrice"),
            col("Quantity"),
            col("TaxPercentage"),
            col("TaxTotal"),
            col("TotalTaxExcl"),
            col("TotalTaxIncl"),
            col("StartDate"),
            col("EndDate"),
            col("Description"),
            col("ProductCode"),
            col("ProductGroupCode")
        )
    )
    
    # Check match rate
    unmatched = line_items_final_sdf.filter(col("InvoiceId").isNull()).count()
    total = line_items_final_sdf.count()
    print(f"  Matched: {total - unmatched:,} / {total:,} ({(total-unmatched)/total*100:.1f}%)")
else:
    # No join key found - add InvoiceId from row_number matching with invoices
    print("\n[WARNING] No join key found - using positional matching with invoices")
    print("  This assumes invoice lines are in the same order as invoices")
    
    # Add row numbers to both dataframes
    line_items_with_row = line_items_transformed_sdf.withColumn("row_num", row_number().over(Window.orderBy(monotonically_increasing_id())))
    invoices_with_row = invoices_final_sdf.select("InvoiceId").withColumn("row_num", row_number().over(Window.orderBy(monotonically_increasing_id())))
    
    # Join on row number
    line_items_final_sdf = (line_items_with_row
        .join(invoices_with_row, "row_num", "left")
        .select(
            col("InvoiceId"),
            col("InvoiceLineId"),
            col("UnitPrice"),
            col("Quantity"),
            col("TaxPercentage"),
            col("TaxTotal"),
            col("TotalTaxExcl"),
            col("TotalTaxIncl"),
            col("StartDate"),
            col("EndDate"),
            col("Description"),
            col("ProductCode"),
            col("ProductGroupCode")
        )
    )
    
    print(f"  Positional match complete")

print(f"\n[SUCCESS] Invoice line items transformed with InvoiceId")

# Repartition for optimal write performance
line_items_final_sdf = line_items_final_sdf.repartition(200)

print(f"\n[SUCCESS] Transformed line items with InvoiceId linkage established")
print("\n" + "="*80)
print("SAMPLE TRANSFORMED LINE ITEMS (First 5 rows)")
print("="*80)
display(line_items_final_sdf.limit(5))

# Verify InvoiceId is present
invoice_id_null_count = line_items_final_sdf.filter(col("InvoiceId").isNull()).count()
print(f"\n[INFO] Line items with null InvoiceId: {invoice_id_null_count:,}")

# Show transformed schema
print("\n" + "="*80)
print("TRANSFORMED LINE ITEMS SCHEMA")
print("="*80)
for field in line_items_final_sdf.schema.fields:
    print(f"  {field.name:30s} : {field.dataType}")

# Load line items to target table
print("\n" + "="*80)
print("LOADING LINE ITEMS DATA")
print("="*80)

try:
    print("\n[INFO] Writing line items to table...")
    print("  Target: teamblue.findata_sandbox.stg_11g_hist_invoice_items")
    print("  Mode: overwrite")
    print("  Partitions: 200")
    print("  Max records per file: 100,000")
    
    (line_items_final_sdf
        .write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("maxRecordsPerFile", 100000)
        .saveAsTable("teamblue.findata_sandbox.stg_11g_hist_invoice_items"))
    
    print("\n[SUCCESS] Line items loaded successfully")
    
    # Verify count
    line_items_loaded_count = spark.table("teamblue.findata_sandbox.stg_11g_hist_invoice_items").count()
    print(f"\n  Records loaded: {line_items_loaded_count:,}")
    
    # Verify data quality
    print("\n" + "="*80)
    print("DATA QUALITY VALIDATION")
    print("="*80)
    
    loaded_table = spark.table("teamblue.findata_sandbox.stg_11g_hist_invoice_items")
    
    # Check for null values in key columns
    key_columns = ["InvoiceId", "InvoiceLineId", "StartDate", "EndDate"]
    for col_name in key_columns:
        null_count = loaded_table.filter(col(col_name).isNull()).count()
        print(f"  {col_name:30s} : {null_count:,} nulls")
    
    # Show statistics
    stats = loaded_table.agg(
        min("StartDate").alias("min_date"),
        max("EndDate").alias("max_date"),
        countDistinct("InvoiceId").alias("distinct_invoices"),
        sum("Quantity").alias("total_quantity")
    ).first()
    
    print(f"\n  Date Range:")
    print(f"    Earliest: {stats['min_date']}")
    print(f"    Latest  : {stats['max_date']}")
    print(f"\n  Distinct Invoices: {stats['distinct_invoices']:,}")
    print(f"  Total Quantity: {stats['total_quantity']:,}")
    
    # Display sample from loaded table
    print("\n" + "="*80)
    print("SAMPLE FROM LOADED TABLE (First 5 rows)")
    print("="*80)
    display(spark.table("teamblue.findata_sandbox.stg_11g_hist_invoice_items").limit(5))
    
    print("\n" + "="*80)
    print("[SUCCESS] Line items data loaded and validated")
    print("="*80)
    
except Exception as e:
    print(f"\n[ERROR] Error loading line items data: {e}")
    import traceback
    traceback.print_exc()
    raise

# Unpersist cached dataframes
invoices_raw_sdf.unpersist()
invoice_lines_raw_sdf.unpersist()
print("\n[INFO] Caches cleared for both invoices and invoice lines")

# COMMAND ----------

print("\n" + "="*80)
print("FINAL DATA QUALITY VALIDATION")
print("="*80)

# Check record counts
invoices_loaded_count = spark.table("teamblue.findata_sandbox.stg_11g_hist_invoices").count()
line_items_loaded_count = spark.table("teamblue.findata_sandbox.stg_11g_hist_invoice_items").count()

print(f"\nRecord Counts:")
print(f"  Invoices loaded: {invoices_loaded_count:,}")
print(f"  Line items loaded: {line_items_loaded_count:,}")

relationship_check = "PASS" if invoices_loaded_count == line_items_loaded_count else "FAIL"
print(f"  Expected 1:1 relationship: {relationship_check}")

# Detailed validation of invoices table
print("\n" + "="*80)
print("INVOICES TABLE VALIDATION")
print("="*80)

invoices_table = spark.table("teamblue.findata_sandbox.stg_11g_hist_invoices")

# Null checks
print("\nNull Value Analysis:")
null_analysis = invoices_table.select([
    count(when(col(c).isNull(), c)).alias(c) 
    for c in invoices_table.columns
])
null_analysis_pd = null_analysis.toPandas().T
null_analysis_pd.columns = ['Null Count']
print(null_analysis_pd[null_analysis_pd['Null Count'] > 0])

# Date range and statistics
date_stats = invoices_table.agg(
    min("InvoiceDate").alias("min_date"),
    max("InvoiceDate").alias("max_date"),
    countDistinct("InvoiceId").alias("distinct_invoices"),
    countDistinct("CustomerKey").alias("distinct_customers"),
    countDistinct("ProviderId").alias("distinct_providers")
).first()

print(f"\nDate Range:")
print(f"  Earliest Invoice: {date_stats['min_date']}")
print(f"  Latest Invoice  : {date_stats['max_date']}")

print(f"\nDistinct Counts:")
print(f"  Invoices : {date_stats['distinct_invoices']:,}")
print(f"  Customers: {date_stats['distinct_customers']:,}")
print(f"  Providers: {date_stats['distinct_providers']:,}")

# Records by year
print("\n" + "="*80)
print("INVOICES BY YEAR")
print("="*80)
year_dist = (invoices_table
    .filter(col("InvoiceDate").isNotNull())
    .withColumn("year", year("InvoiceDate"))
    .groupBy("year")
    .agg(
        count("*").alias("invoice_count"),
        countDistinct("CustomerKey").alias("unique_customers")
    )
    .orderBy("year"))

display(year_dist)

# Top countries
print("\n" + "="*80)
print("TOP 10 COUNTRIES BY INVOICE COUNT")
print("="*80)
country_dist = (invoices_table
    .filter(col("BillingCountryCode").isNotNull())
    .groupBy("BillingCountryCode")
    .agg(count("*").alias("invoice_count"))
    .orderBy(col("invoice_count").desc())
    .limit(10))

display(country_dist)

# Display sample data from invoices
print("\n" + "="*80)
print("SAMPLE FROM INVOICES TABLE")
print("="*80)
display(invoices_table.limit(5))

# Detailed validation of line items table
print("\n" + "="*80)
print("LINE ITEMS TABLE VALIDATION")
print("="*80)

line_items_table = spark.table("teamblue.findata_sandbox.stg_11g_hist_invoice_items")

# Statistics
line_stats = line_items_table.agg(
    countDistinct("InvoiceId").alias("distinct_invoices"),
    sum("Quantity").alias("total_quantity"),
    avg("TaxPercentage").alias("avg_tax_pct"),
    min("StartDate").alias("min_date"),
    max("EndDate").alias("max_date")
).first()

print(f"\nLine Items Statistics:")
print(f"  Distinct Invoices: {line_stats['distinct_invoices']:,}")
print(f"  Total Quantity   : {line_stats['total_quantity']:,}")
print(f"  Avg Tax %        : {line_stats['avg_tax_pct']:.2f}%")
print(f"  Date Range       : {line_stats['min_date']} to {line_stats['max_date']}")

# Display sample data from line items
print("\n" + "="*80)
print("SAMPLE FROM LINE ITEMS TABLE")
print("="*80)
display(line_items_table.limit(5))

# Verify referential integrity
print("\n" + "="*80)
print("REFERENTIAL INTEGRITY CHECK")
print("="*80)

print("\n[INFO] Checking Invoice-to-LineItem relationship...")

# Check for invoices without line items
invoices_without_items = (invoices_table
    .join(line_items_table, "InvoiceId", "left_anti")
    .count())

# Check for line items without invoices
items_without_invoices = (line_items_table
    .join(invoices_table, "InvoiceId", "left_anti")
    .count())

print(f"  Invoices without line items: {invoices_without_items:,}")
print(f"  Line items without invoices: {items_without_invoices:,}")

# Calculate match statistics
total_invoices = invoices_table.count()
total_line_items = line_items_table.count()
invoices_with_items = total_invoices - invoices_without_items
items_with_invoices = total_line_items - items_without_invoices

match_rate_invoices = (invoices_with_items / total_invoices * 100) if total_invoices > 0 else 0
match_rate_items = (items_with_invoices / total_line_items * 100) if total_line_items > 0 else 0

print(f"\n  Match Statistics:")
print(f"    Invoices with line items: {invoices_with_items:,} ({match_rate_invoices:.2f}%)")
print(f"    Line items with invoices: {items_with_invoices:,} ({match_rate_items:.2f}%)")

integrity_check = "PASS" if (invoices_without_items == 0 and items_without_invoices == 0) else "WARNING" if items_without_invoices == 0 else "FAIL"
print(f"\n  Referential Integrity Status: {integrity_check}")

if integrity_check == "FAIL":
    print(f"    [ERROR] Found {items_without_invoices:,} orphaned line items!")
elif integrity_check == "WARNING":
    print(f"    [WARNING] Found {invoices_without_items:,} invoices without line items")
else:
    print(f"    [SUCCESS] All relationships properly established")

# Pipeline execution summary
print("\n" + "="*80)
print("11G HISTORICAL INVOICES ETL PIPELINE - FINAL SUMMARY")
print("="*80)

print(f"\nDATA PROCESSING RESULTS:")
print(f"  Invoices source file    : COM_20251023_144040_invoices.csv")
print(f"  Line items source file  : COM_20251023_144040_invoicelines.csv")
print(f"  Invoices loaded         : {invoices_loaded_count:,}")
print(f"  Line items loaded       : {line_items_loaded_count:,}")
print(f"  Data quality            : {relationship_check}")
print(f"  Referential integrity   : {integrity_check}")

print(f"\nTARGET TABLES:")
print(f"  teamblue.findata_sandbox.stg_11g_hist_invoices")
print(f"  teamblue.findata_sandbox.stg_11g_hist_invoice_items")

print(f"\nTRANSFORMATIONS APPLIED:")
print(f"  Date parsing: d/MM/yyyy format to DATE type")
print(f"  European decimal format: comma to dot conversion")
print(f"  Data type casting: String to Long/Double")
print(f"  String cleaning: removed quotes and newlines")
print(f"  Encoding fix: handled UTF-16 BOM in column names")
print(f"  Added df_source field to invoices")
print(f"  Loaded invoice line items from dedicated CSV file")
print(f"  Established InvoiceId linkage via join (if not present in source)")
print(f"  Verified referential integrity between invoices and line items")

print(f"\nDATA CHARACTERISTICS:")
print(f"  Invoices date range : {date_stats['min_date']} to {date_stats['max_date']}")
print(f"  Unique customers    : {date_stats['distinct_customers']:,}")
print(f"  Unique providers    : {date_stats['distinct_providers']:,}")
print(f"  Invoice-to-line-item relationship validation: {relationship_check}")

print(f"\nPIPELINE STATUS: COMPLETED SUCCESSFULLY")
print("="*80)

# COMMAND ----------

