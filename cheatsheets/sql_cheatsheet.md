# SQL Cheatsheet (Institutional Data/AI Lab)

## Short Simplified Definitions

- `SELECT`: Choose columns to return.
- `WHERE`: Filter rows before grouping/aggregation.
- `GROUP BY`: Combine rows into groups for summaries.
- `HAVING`: Filter groups after aggregation.

## Core Query Patterns

```sql
SELECT dep_airport, arr_airport, AVG(delay_minutes) AS avg_delay
FROM flights
WHERE scheduled_dep >= '2026-01-01'
GROUP BY dep_airport, arr_airport
HAVING COUNT(*) > 10
ORDER BY avg_delay DESC;
```

## JOIN Types

- `INNER JOIN`: Rows that match in both tables
- `LEFT JOIN`: All rows from left table + matches from right
- `RIGHT JOIN`: All rows from right table + matches from left
- `FULL OUTER JOIN`: All rows from both tables, matched where possible

## Index Basics

- Add indexes to columns used often in `WHERE`, `JOIN`, `ORDER BY`
- Indexes speed reads but add write overhead
- Too many indexes can slow ingestion and updates

## EXPLAIN Concept

- `EXPLAIN` shows the query execution plan
- Use it to see scans, joins, and index usage
- `EXPLAIN ANALYZE` runs the query and shows actual timing (use carefully on large tables)

## Common Pitfalls

- `SELECT *` in production queries (returns unnecessary data)
- Missing filters on large tables
- Using functions on indexed columns in ways that block index usage
- Ignoring `NULL` handling in comparisons and aggregates
- Joining on non-unique keys and accidentally duplicating rows
