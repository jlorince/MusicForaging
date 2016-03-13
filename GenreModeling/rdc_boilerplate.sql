# Misc. queries for pulling data from RDC MySQL database

# Usage:
# mysql --defaults-file=/N/u/jlorince/RDC/.my.cnf -D analysis_lastfm -N < extractData.sql > OUTPUT_PATH

# boilerplate for working with RDC
SET SQL_BIG_SELECTS=1;SET time_zone = '+00:00';set sql_select_limit=18446744073709551615;SET group_concat_max_len=18446744073709551615;

