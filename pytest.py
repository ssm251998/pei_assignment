import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

TABLES = [
    "raw_orders", "raw_products", "raw_customers", "master_sales",
    "profit_by_year", "profit_by_year_category",
    "profit_by_customer", "profit_by_customer_year",
]

@pytest.fixture(scope="session")
def spark():
    session = SparkSession.getActiveSession()
    if session is not None:
        return session
    return (SparkSession.builder
            .appName("pytest")
            .master("local[2]")
            .getOrCreate())

@pytest.fixture(scope="session")
def sales_tables(spark):
    missing = [t for t in TABLES if not spark.catalog.tableExists(t)]
    if missing:
        pytest.fail("Missing tables: " + ", ".join(missing) +
                    ". Run the transformation notebook first.")
    return {t: spark.table(t) for t in TABLES}

def assert_columns(df, expected, table_name):
    missing = sorted(set(expected) - set(df.columns))
    assert not missing, f"{table_name} is missing columns: {missing}"

def test_raw_tables_not_empty(sales_tables):
    for name in ["raw_orders", "raw_products", "raw_customers"]:
        assert sales_tables[name].limit(1).count() > 0

def test_orders_schema(sales_tables):
    assert_columns(sales_tables["raw_orders"], [
        "Row_ID", "Order_ID", "Order_Date", "Ship_Date", "Ship_Mode",
        "Customer_ID", "Product_ID", "Quantity", "Price", "Discount", "Profit"
    ], "raw_orders")

def test_products_schema(sales_tables):
    assert_columns(sales_tables["raw_products"], [
        "Product_ID", "Category", "Sub_Category", "Product_Name",
        "State", "Product_Price"
    ], "raw_products")

def test_customers_schema(sales_tables):
    assert_columns(sales_tables["raw_customers"], [
        "Customer_ID", "Customer_Name", "Segment", "Country", "City",
        "State", "Postal_Code", "Region"
    ], "raw_customers")

def test_master_sales_schema(sales_tables):
    assert_columns(sales_tables["master_sales"], [
        "Row_ID", "Order_ID", "Order_Date", "Ship_Date", "Ship_Mode",
        "Customer_ID", "Customer_Name", "Segment", "Customer_Country",
        "Customer_City", "Customer_State", "Postal_Code", "Region",
        "Product_ID", "Product_Name", "Category", "Sub_Category",
        "Product_State", "Product_Price", "Quantity", "Price", "Discount",
        "Profit", "Year"
    ], "master_sales")

@pytest.mark.parametrize(
    "table_name,key",
    [
        ("raw_orders", "Row_ID"),
        ("raw_products", "Product_ID"),
        ("raw_customers", "Customer_ID"),
        ("master_sales", "Row_ID"),
    ],
)
def test_keys_are_unique(sales_tables, table_name, key):
    duplicates = (sales_tables[table_name].groupBy(key).count() q
                  .filter(F.col("count") > 1).limit(1).count())
    assert duplicates == 0

def test_master_row_count_matches_orders(sales_tables):
    assert sales_tables["master_sales"].count() == sales_tables["raw_orders"].count()

def test_required_master_columns_have_no_nulls(sales_tables):
    df = sales_tables["master_sales"]
    required = [
        "Row_ID", "Order_ID", "Order_Date", "Customer_ID", "Product_ID",
        "Quantity", "Price", "Discount", "Profit", "Year"
    ]
    condition = None
    for column in required:
        current = F.col(column).isNull()
        condition = current if condition is None else condition | current
    assert df.filter(condition).limit(1).count() == 0

def test_quantity_is_positive(sales_tables):
    assert sales_tables["master_sales"].filter(F.col("Quantity") <= 0).limit(1).count() == 0

def test_discount_is_between_zero_and_one(sales_tables):
    invalid = sales_tables["master_sales"].filter(
        (F.col("Discount") < 0) | (F.col("Discount") > 1)
    )
    assert invalid.limit(1).count() == 0

def test_customer_join_is_complete(sales_tables):
    assert sales_tables["master_sales"].filter(
        F.col("Customer_Name").isNull()
    ).limit(1).count() == 0

def test_product_join_is_complete(sales_tables):
    assert sales_tables["master_sales"].filter(
        F.col("Category").isNull()
    ).limit(1).count() == 0

@pytest.mark.parametrize(
    "table_name,columns",
    [
        ("profit_by_year", ["Year", "Total_Profit"]),
        ("profit_by_year_category", ["Year", "Category", "Total_Profit"]),
        ("profit_by_customer", ["Customer_ID", "Customer_Name", "Total_Profit"]),
        ("profit_by_customer_year",
         ["Customer_ID", "Customer_Name", "Year", "Total_Profit"]),
    ],
)
def test_aggregate_schemas(sales_tables, table_name, columns):
    assert_columns(sales_tables[table_name], columns, table_name)

@pytest.mark.parametrize(
    "table_name,group_columns",
    [
        ("profit_by_year", ["Year"]),
        ("profit_by_year_category", ["Year", "Category"]),
        ("profit_by_customer", ["Customer_ID", "Customer_Name"]),
        ("profit_by_customer_year",
         ["Customer_ID", "Customer_Name", "Year"]),
    ],
)
def test_aggregate_group_keys_are_unique(sales_tables, table_name, group_columns):
    duplicates = (sales_tables[table_name].groupBy(*group_columns).count()
                  .filter(F.col("count") > 1).limit(1).count())
    assert duplicates == 0

def test_profit_by_year_reconciles(sales_tables):
    master = sales_tables["master_sales"].agg(F.sum("Profit").alias("total")).first()["total"]
    yearly = sales_tables["profit_by_year"].agg(F.sum("Total_Profit").alias("total")).first()["total"]
    assert float(yearly or 0) == pytest.approx(float(master or 0), abs=0.01)

@pytest.mark.parametrize(
    "table_name",
    ["profit_by_year_category", "profit_by_customer", "profit_by_customer_year"],
)
def test_profit_aggregates_reconcile(sales_tables, table_name):
    master = sales_tables["master_sales"].agg(F.sum("Profit").alias("total")).first()["total"]
    aggregate = sales_tables[table_name].agg(F.sum("Total_Profit").alias("total")).first()["total"]
    assert float(aggregate or 0) == pytest.approx(float(master or 0), abs=0.01)
