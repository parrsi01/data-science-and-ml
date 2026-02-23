# Databases and Data Engineering

---

> **Field** — Data Engineering and Database Systems
> **Scope** — Relational database concepts, SQL
> fundamentals, ETL/ELT patterns, and data pipeline
> architecture for data science workflows

---

## Overview

Data science starts with data, and most
real-world data lives in databases. Before
you can build models or run analyses, you
need to know how to store, query, transform,
and move data efficiently. This reference
covers the database and data engineering
concepts that every data scientist encounters
when working with production systems.

---

## Definitions

---

### `Schema`

**Definition.**
A schema is the structure or blueprint of a
database. It defines what tables exist, what
columns each table has, what data types those
columns use, and how the tables relate to
each other.

**Context.**
Understanding a database schema is the first
step in any data science project that uses
a relational database. The schema tells you
what data is available, how it is organized,
and how different pieces of information
connect. When you join tables, filter rows,
or aggregate data, you are working within
the constraints defined by the schema.

**Example.**
A simple e-commerce schema:

```sql
CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    email         TEXT UNIQUE,
    created_at    TIMESTAMP
);

CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER REFERENCES customers,
    total_amount  DECIMAL(10, 2),
    order_date    DATE
);
```

This schema defines two tables and their
relationship: each order belongs to a
customer via the `customer_id` column.

---

### `Primary Key`

**Definition.**
A primary key is a column (or set of columns)
that uniquely identifies each row in a table.
No two rows can have the same primary key
value, and the value cannot be null.

**Context.**
Primary keys are the backbone of relational
databases. They ensure every record is unique
and can be referenced precisely. When you
look up a specific customer, order, or
product, you use the primary key. They are
also what foreign keys point to when linking
tables together.

**Example.**
```sql
CREATE TABLE products (
    product_id  INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    price       DECIMAL(8, 2)
);

-- Each product has a unique product_id
-- You can always find exactly one row:
SELECT * FROM products
WHERE product_id = 42;
```

Common primary key patterns:

- **Auto-increment integer:**
  1, 2, 3, 4, ... (most common)
- **UUID:**
  'a1b2c3d4-...' (globally unique)
- **Composite key:**
  Two or more columns together
  (e.g., student_id + course_id)

---

### `Foreign Key`

**Definition.**
A foreign key is a column in one table that
references the primary key of another table.
It creates a link between the two tables,
enforcing that the referenced value must
exist.

**Context.**
Foreign keys define relationships between
tables. They are what make SQL JOINs work.
When you join an orders table to a customers
table, the foreign key (customer_id in
orders) tells the database which customer
each order belongs to. Foreign keys also
prevent orphaned records (e.g., an order
that points to a non-existent customer).

**Example.**
```sql
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER,
    total        DECIMAL(10, 2),
    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);

-- This will succeed (customer 5 exists):
INSERT INTO orders VALUES (1, 5, 99.99);

-- This will FAIL (customer 999 does not exist):
INSERT INTO orders VALUES (2, 999, 49.99);
-- Error: foreign key constraint violation
```

The foreign key guarantees data integrity:
every order must belong to a real customer.

---

### `Index (database)`

**Definition.**
A database index is a data structure that
speeds up the retrieval of rows from a table.
It works like the index at the back of a
book: instead of scanning every page, you
look up a topic and jump directly to the
right page.

**Context.**
Without an index, the database must scan
every row in the table to find matching
records (called a "full table scan"). For
small tables this is fine, but for tables
with millions of rows, queries can take
minutes instead of milliseconds. Indexes
make queries fast but use extra disk space
and slow down writes (inserts/updates),
so you create them strategically on columns
that are frequently searched or filtered.

**Example.**
```sql
-- Without an index, this scans every row:
SELECT * FROM orders
WHERE customer_id = 42;

-- Create an index on customer_id:
CREATE INDEX idx_orders_customer
ON orders(customer_id);

-- Now the same query uses the index
-- and returns results much faster.
```

When to create indexes:

- Columns used in WHERE clauses
- Columns used in JOIN conditions
- Columns used in ORDER BY
- Columns with high cardinality
  (many distinct values)

When NOT to create indexes:

- Tables with very few rows
- Columns that are rarely queried
- Tables with heavy insert/update load

---

### `SQL SELECT`

**Definition.**
SELECT is the SQL statement used to retrieve
data from one or more tables. It is the most
fundamental and frequently used SQL command.

**Context.**
Every data science query starts with SELECT.
Whether you are exploring data, building
features, or generating reports, you use
SELECT to pull the data you need. Mastering
SELECT (with its clauses like WHERE,
GROUP BY, ORDER BY, and LIMIT) is the most
important SQL skill for a data scientist.

**Example.**
```sql
-- Select all columns from a table
SELECT * FROM customers;

-- Select specific columns
SELECT name, email FROM customers;

-- With aliases for readability
SELECT
    name         AS customer_name,
    email        AS contact_email,
    created_at   AS signup_date
FROM customers;

-- Limit results (useful for exploration)
SELECT * FROM customers LIMIT 10;

-- Count rows
SELECT COUNT(*) FROM customers;
```

Key clauses (in execution order):

1. FROM — which table(s)
2. WHERE — filter rows
3. GROUP BY — aggregate groups
4. HAVING — filter groups
5. SELECT — choose columns
6. ORDER BY — sort results
7. LIMIT — restrict output rows

---

### `SQL WHERE`

**Definition.**
The WHERE clause filters rows based on a
condition. Only rows that satisfy the
condition are included in the result.

**Context.**
WHERE is how you narrow down your data.
Instead of pulling every row from a
million-row table, you use WHERE to get
only the rows you need. This is essential
for both data exploration and building
efficient queries that do not waste memory
or time processing irrelevant data.

**Example.**
```sql
-- Simple comparison
SELECT * FROM orders
WHERE total_amount > 100;

-- Multiple conditions with AND
SELECT * FROM orders
WHERE total_amount > 100
  AND order_date >= '2024-01-01';

-- OR condition
SELECT * FROM customers
WHERE country = 'US'
   OR country = 'UK';

-- IN for multiple values
SELECT * FROM customers
WHERE country IN ('US', 'UK', 'CA');

-- Pattern matching with LIKE
SELECT * FROM customers
WHERE email LIKE '%@gmail.com';

-- NULL checking
SELECT * FROM customers
WHERE phone IS NULL;

-- NOT NULL
SELECT * FROM customers
WHERE phone IS NOT NULL;
```

Common operators:

- `=`, `!=`, `<`, `>`, `<=`, `>=`
- `AND`, `OR`, `NOT`
- `IN`, `NOT IN`
- `LIKE`, `ILIKE` (case-insensitive)
- `IS NULL`, `IS NOT NULL`
- `BETWEEN`

---

### `SQL GROUP BY`

**Definition.**
GROUP BY groups rows that share the same
value in one or more columns and lets you
apply aggregate functions (COUNT, SUM, AVG,
MIN, MAX) to each group.

**Context.**
GROUP BY is essential for summarizing data.
In data science, you use it constantly:
counting events per user, averaging sales
per region, finding the maximum temperature
per city. It turns raw rows into meaningful
summaries. If you can think of it as
"calculate X per Y," you probably need
GROUP BY.

**Example.**
```sql
-- Count orders per customer
SELECT
    customer_id,
    COUNT(*) AS order_count
FROM orders
GROUP BY customer_id;

-- Average order amount per month
SELECT
    DATE_TRUNC('month', order_date) AS month,
    AVG(total_amount)               AS avg_amount,
    COUNT(*)                        AS num_orders
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

-- Filter groups with HAVING
SELECT
    customer_id,
    SUM(total_amount) AS total_spent
FROM orders
GROUP BY customer_id
HAVING SUM(total_amount) > 1000;
```

Important distinction:

- **WHERE** filters individual rows
  (before grouping)
- **HAVING** filters groups
  (after grouping)

---

### `SQL JOIN`

**Definition.**
A JOIN combines rows from two or more tables
based on a related column. It lets you pull
data from multiple tables in a single query.

**Context.**
Real-world data is almost always spread
across multiple tables (customers, orders,
products, etc.). JOINs are how you bring
this data together. Understanding JOIN types
is critical for data science because
incorrect joins can silently duplicate or
drop rows, leading to wrong analyses.

**Example.**
```sql
-- INNER JOIN: only matching rows
SELECT
    c.name,
    o.order_id,
    o.total_amount
FROM customers c
INNER JOIN orders o
    ON c.customer_id = o.customer_id;

-- LEFT JOIN: all customers,
-- even those with no orders
SELECT
    c.name,
    COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
GROUP BY c.name;
```

JOIN types:

- **INNER JOIN:**
  Only rows that match in both tables.

- **LEFT JOIN:**
  All rows from left table.
  NULLs where right table has no match.

- **RIGHT JOIN:**
  All rows from right table.
  NULLs where left table has no match.

- **FULL OUTER JOIN:**
  All rows from both tables.
  NULLs where either side has no match.

- **CROSS JOIN:**
  Every row paired with every other row.
  Rarely used. Produces huge results.

---

### `EXPLAIN Query Plan`

**Definition.**
EXPLAIN is a SQL command that shows how the
database plans to execute your query. It
reveals whether the database will use
indexes, perform full table scans, or use
other strategies to retrieve your data.

**Context.**
When a query is slow, EXPLAIN tells you
why. It is the primary tool for SQL
performance debugging. Data scientists
working with large production databases
need to understand query plans to avoid
queries that take hours instead of seconds.
If EXPLAIN shows a "Seq Scan" on a
million-row table, you probably need an
index.

**Example.**
```sql
-- Basic explain
EXPLAIN SELECT * FROM orders
WHERE customer_id = 42;

-- With execution timing and details
EXPLAIN ANALYZE SELECT * FROM orders
WHERE customer_id = 42;
```

Sample output:

```
Index Scan using idx_orders_customer
  on orders
  Index Cond: (customer_id = 42)
  Planning Time: 0.1 ms
  Execution Time: 0.05 ms
```

What to look for:

- **Seq Scan:** Full table scan (slow
  on large tables)
- **Index Scan:** Uses an index (fast)
- **Hash Join / Merge Join:** How tables
  are joined
- **Cost:** Estimated computational cost
- **Rows:** Estimated number of rows

---

### `ETL vs ELT`

**Definition.**
ETL (Extract, Transform, Load) and ELT
(Extract, Load, Transform) are two patterns
for moving data from source systems into
a data warehouse. In ETL, data is
transformed before loading. In ELT, raw
data is loaded first and transformed inside
the warehouse.

**Context.**
ETL is the traditional approach where data
is cleaned and structured before it enters
the warehouse. ELT has become more popular
with modern cloud warehouses (Snowflake,
BigQuery, Redshift) that have enough
computing power to handle transformations
after loading. Data scientists need to
understand both because the pattern used
determines where and how you access clean
data.

**Example.**
**ETL workflow:**

1. **Extract:** Pull data from source
   (API, database, files)
2. **Transform:** Clean, validate, reshape
   in a staging area (Python, Spark)
3. **Load:** Insert clean data into the
   warehouse

**ELT workflow:**

1. **Extract:** Pull raw data from source
2. **Load:** Dump raw data into the
   warehouse as-is
3. **Transform:** Use SQL inside the
   warehouse to clean and reshape

```
ETL:  Source -> [Transform] -> Warehouse
ELT:  Source -> Warehouse -> [Transform]
```

When to use each:

- **ETL:** sensitive data that needs
  redaction before loading, legacy systems
- **ELT:** modern cloud warehouses,
  when you want raw data available for
  ad-hoc queries

---

### `SQLAlchemy`

**Definition.**
SQLAlchemy is a Python library for working
with relational databases. It provides both
a low-level SQL toolkit and a high-level
ORM (Object-Relational Mapper) that lets
you interact with databases using Python
objects instead of raw SQL strings.

**Context.**
SQLAlchemy is the standard way to connect
Python data science code to databases. It
integrates with Pandas (`pd.read_sql()`),
works with virtually any database (SQLite,
PostgreSQL, MySQL, etc.), and handles
connection pooling, transactions, and
security (SQL injection prevention). Most
production data pipelines use SQLAlchemy
for database interactions.

**Example.**
```python
from sqlalchemy import create_engine
import pandas as pd

# Connect to a PostgreSQL database
engine = create_engine(
    'postgresql://user:pass@host:5432/mydb'
)

# Read data into a Pandas DataFrame
df = pd.read_sql(
    "SELECT * FROM customers WHERE age > 25",
    engine
)

# Write a DataFrame to a table
df.to_sql(
    'results',
    engine,
    if_exists='replace',
    index=False
)
```

Connection string formats:

- SQLite: `sqlite:///mydata.db`
- PostgreSQL: `postgresql://user:pass@host/db`
- MySQL: `mysql://user:pass@host/db`

---

### `PostgreSQL`

**Definition.**
PostgreSQL (often called "Postgres") is a
powerful, open-source relational database
management system. It is known for its
reliability, feature richness, and strong
SQL compliance.

**Context.**
PostgreSQL is one of the most popular
databases in the data science ecosystem.
It supports advanced features like JSON
columns, full-text search, window functions,
and array types that are particularly useful
for analytical queries. Many data science
teams use PostgreSQL as their primary
production database and analytical store.

**Example.**
Connecting from the command line:

```bash
# Connect to a database
psql -h localhost -U myuser -d mydb

# List all tables
\dt

# Describe a table's structure
\d customers

# Run a query
SELECT * FROM customers LIMIT 5;

# Exit
\q
```

Connecting from Python:

```python
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine(
    'postgresql://user:pass@localhost:5432/mydb'
)
df = pd.read_sql('SELECT * FROM orders', engine)
```

PostgreSQL features useful for data science:

- Window functions (LAG, LEAD, RANK)
- JSON/JSONB columns for semi-structured data
- CTEs (Common Table Expressions)
- COPY for fast bulk loading

---

### `Batch Processing`

**Definition.**
Batch processing is a data processing
pattern where data is collected over a
period of time and then processed all at
once in a single job, rather than being
processed immediately as it arrives.

**Context.**
Most data science workloads use batch
processing. Training a model on yesterday's
data, generating a daily report, or running
a nightly ETL job are all batch processes.
The alternative is stream processing, where
data is processed in real-time as it
arrives. Batch processing is simpler to
implement, easier to debug, and suitable
for most data science use cases.

**Example.**
Common batch processing pattern:

```python
import pandas as pd
from pathlib import Path

def process_daily_batch(date_str):
    """Process all data for a given date."""
    # 1. Read the day's data
    path = Path(f'data/raw/{date_str}.csv')
    df = pd.read_csv(path)

    # 2. Transform
    df['amount'] = df['amount'].abs()
    df = df.dropna(subset=['customer_id'])

    # 3. Aggregate
    summary = df.groupby('category').agg(
        total=('amount', 'sum'),
        count=('amount', 'count')
    )

    # 4. Save results
    out = Path(f'data/processed/{date_str}.csv')
    summary.to_csv(out)

process_daily_batch('2024-01-15')
```

Batch vs Stream:

- **Batch:** Process data in chunks
  (hourly, daily). Simple. High latency.
- **Stream:** Process data as it arrives.
  Complex. Low latency.

---

### `Data Pipeline`

**Definition.**
A data pipeline is an automated sequence of
steps that moves data from one or more
sources, transforms it, and delivers it to
a destination (database, file, dashboard,
or model). Each step depends on the previous
one completing successfully.

**Context.**
Data pipelines are the infrastructure that
keeps data science running. Without them,
data scientists would spend all their time
manually downloading, cleaning, and loading
data. A well-built pipeline runs
automatically on a schedule, handles errors
gracefully, and ensures data is fresh and
reliable when the data scientist needs it.

**Example.**
A simple pipeline structure:

```
[Data Sources]
     |
     v
[Extract] -- Pull from APIs, databases, files
     |
     v
[Validate] -- Check schema, nulls, ranges
     |
     v
[Transform] -- Clean, enrich, aggregate
     |
     v
[Load] -- Write to warehouse or data lake
     |
     v
[Report/Model] -- Dashboards, ML training
```

Pipeline tools commonly used:

- **Airflow:** schedule and monitor workflows
- **Prefect:** modern Python-native pipelines
- **dbt:** SQL-based transformations
- **Luigi:** simple Python task pipelines
- **Cron:** basic scheduling (Linux)

Key properties of good pipelines:

- **Idempotent:** running twice produces
  the same result
- **Observable:** logs, metrics, alerts
- **Recoverable:** can restart from failure

---

## See Also

- [Python and Numerical Computing](./02_python_and_numerical_computing.md)
- [Data Quality and Validation](./04_data_quality_and_validation.md)
- [Machine Learning Fundamentals](./05_machine_learning_fundamentals.md)

---

> **Author** — Simon Parris | Data Science Reference Library
