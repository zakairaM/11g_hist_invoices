# Databricks notebook source
# ============================================================================
# Check View Definition Access in Databricks
# ============================================================================
# This notebook helps you verify if you have access to view definitions
# in Databricks before creating or working with views

from pyspark.sql.functions import *

# COMMAND ----------

# ============================================================================
# METHOD 1: Check if you can see the definition of an existing view
# ============================================================================

print("="*80)
print("METHOD 1: Testing SHOW CREATE VIEW on an existing view")
print("="*80)

# Replace with an actual view name in your database
TEST_VIEW_NAME = "teamblue.findata_sandbox.some_existing_view"

try:
    # Try to get the view definition
    result = spark.sql(f"SHOW CREATE TABLE {TEST_VIEW_NAME}")
    print(f"\n✓ SUCCESS: You have access to view definitions!")
    print(f"\nView definition for {TEST_VIEW_NAME}:")
    result.show(truncate=False)
except Exception as e:
    print(f"\n✗ FAILED: Cannot access view definition")
    print(f"Error: {e}")
    print("\nThis could mean:")
    print("  - The view doesn't exist")
    print("  - You don't have permissions to see the view definition")
    print("  - You only have SELECT privilege but not metadata access")

# COMMAND ----------

# ============================================================================
# METHOD 2: Check permissions on a specific view
# ============================================================================

print("="*80)
print("METHOD 2: Checking your permissions on a view")
print("="*80)

TEST_VIEW_NAME = "teamblue.findata_sandbox.some_existing_view"

try:
    # Show grants on the view
    result = spark.sql(f"SHOW GRANT ON TABLE {TEST_VIEW_NAME}")
    print(f"\n✓ SUCCESS: Retrieved permissions for {TEST_VIEW_NAME}")
    result.show(truncate=False)
    
    print("\nLook for these privileges:")
    print("  - SELECT: Can query the view")
    print("  - MODIFY: Can alter the view")
    print("  - READ_METADATA: Can see view definition")
    print("  - ALL PRIVILEGES: Full access including viewing definition")
    
except Exception as e:
    print(f"\n✗ FAILED: Cannot retrieve permissions")
    print(f"Error: {e}")
    print("\nYou may not have permission to view grants on this view")

# COMMAND ----------

# ============================================================================
# METHOD 3: List all views you can see in a schema
# ============================================================================

print("="*80)
print("METHOD 3: Listing all views in a schema")
print("="*80)

SCHEMA_NAME = "teamblue.findata_sandbox"

try:
    # Show all views in the schema
    result = spark.sql(f"SHOW VIEWS IN {SCHEMA_NAME}")
    print(f"\n✓ SUCCESS: You can see views in {SCHEMA_NAME}")
    result.show(truncate=False)
    
    view_count = result.count()
    print(f"\nFound {view_count} view(s) in {SCHEMA_NAME}")
    
except Exception as e:
    print(f"\n✗ FAILED: Cannot list views")
    print(f"Error: {e}")

# COMMAND ----------

# ============================================================================
# METHOD 4: Check detailed metadata for a view
# ============================================================================

print("="*80)
print("METHOD 4: Getting detailed view metadata")
print("="*80)

TEST_VIEW_NAME = "teamblue.findata_sandbox.some_existing_view"

try:
    # Get extended description including view definition
    result = spark.sql(f"DESCRIBE EXTENDED {TEST_VIEW_NAME}")
    print(f"\n✓ SUCCESS: Retrieved metadata for {TEST_VIEW_NAME}")
    result.show(truncate=False)
    
    # Look for the "View Text" or "View Definition" in the output
    view_text_rows = result.filter(col("col_name") == "View Text").collect()
    if view_text_rows:
        print("\n✓ View definition is available!")
        print("View Text:", view_text_rows[0]["data_type"])
    else:
        print("\n⚠ WARNING: Metadata retrieved but view definition not shown")
        print("You may have SELECT but not READ_METADATA privilege")
    
except Exception as e:
    print(f"\n✗ FAILED: Cannot access view metadata")
    print(f"Error: {e}")

# COMMAND ----------

# ============================================================================
# METHOD 5: Test creating a simple view
# ============================================================================

print("="*80)
print("METHOD 5: Testing view creation permissions")
print("="*80)

TEST_TABLE = "teamblue.findata_sandbox.stg_11g_hist_invoices"
TEST_VIEW = "teamblue.findata_sandbox.test_view_access_check"

try:
    # Try to create a simple view
    spark.sql(f"""
        CREATE OR REPLACE VIEW {TEST_VIEW} AS
        SELECT InvoiceId, InvoiceNumber, InvoiceDate
        FROM {TEST_TABLE}
        LIMIT 10
    """)
    print(f"\n✓ SUCCESS: You can create views in this schema!")
    
    # Now check if you can see its definition
    definition = spark.sql(f"SHOW CREATE TABLE {TEST_VIEW}")
    print(f"\n✓ SUCCESS: You can also see the view definition you created!")
    definition.show(truncate=False)
    
    # Clean up the test view
    spark.sql(f"DROP VIEW IF EXISTS {TEST_VIEW}")
    print(f"\n✓ Test view cleaned up")
    
except Exception as e:
    print(f"\n✗ FAILED: Cannot create view or access its definition")
    print(f"Error: {e}")
    print("\nPossible reasons:")
    print("  - You don't have CREATE VIEW privilege on this schema")
    print("  - You don't have SELECT privilege on the source table")
    print("  - Schema doesn't allow view creation")

# COMMAND ----------

# ============================================================================
# METHOD 6: Check your current user and privileges
# ============================================================================

print("="*80)
print("METHOD 6: Checking your user identity and privileges")
print("="*80)

try:
    # Get current user
    current_user = spark.sql("SELECT current_user() as user").collect()[0]["user"]
    print(f"\nCurrent user: {current_user}")
    
    # Try to see all privileges on the schema
    SCHEMA_NAME = "teamblue.findata_sandbox"
    try:
        schema_privileges = spark.sql(f"SHOW GRANT ON SCHEMA {SCHEMA_NAME}")
        print(f"\nPrivileges on schema {SCHEMA_NAME}:")
        schema_privileges.show(truncate=False)
    except Exception as e:
        print(f"\nCannot retrieve schema privileges: {e}")
    
except Exception as e:
    print(f"\n✗ FAILED: Cannot get user info")
    print(f"Error: {e}")

# COMMAND ----------

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: How to Check View Definition Access")
print("="*80)

print("""
To check if you have access to view definitions in Databricks, you need:

1. **READ_METADATA or ALL PRIVILEGES** on the view/schema
   - This allows you to see the view's SQL definition
   - Without this, you can only query the view but not see how it's built

2. **Key Commands to Test:**
   
   a) See view definition:
      SHOW CREATE TABLE <view_name>
   
   b) Check your permissions:
      SHOW GRANT ON TABLE <view_name>
   
   c) Get detailed metadata:
      DESCRIBE EXTENDED <view_name>
   
   d) List available views:
      SHOW VIEWS IN <schema_name>

3. **What You Need to Create a View:**
   - CREATE privilege on the target schema
   - SELECT privilege on all source tables/views
   - READ_METADATA to see existing view definitions (optional)

4. **Quick Test:**
   Run this command with an existing view name:
   
   SHOW CREATE TABLE teamblue.findata_sandbox.<some_view_name>
   
   If it works → You have access to view definitions ✓
   If it fails → You may only have SELECT privilege

5. **In Databricks UI:**
   - Go to Data Explorer
   - Find your schema: teamblue.findata_sandbox
   - Click on any existing view
   - Look for a "Definition" or "DDL" tab
   - If you can see the SQL code → You have access ✓

6. **Ask Your Admin If You Need:**
   - GRANT READ_METADATA ON VIEW <view_name> TO `<your_user>`
   - Or: GRANT CREATE ON SCHEMA <schema_name> TO `<your_user>`
""")

print("="*80)
print("END OF ACCESS CHECK")
print("="*80)

# COMMAND ----------
