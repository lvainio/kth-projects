/* ----- BOOKS ----- */
DROP TABLE IF EXISTS book           CASCADE;
DROP TABLE IF EXISTS author         CASCADE;
DROP TABLE IF EXISTS book_author    CASCADE;
DROP TABLE IF EXISTS genre          CASCADE;
DROP TABLE IF EXISTS book_genre     CASCADE;

/* ----- RELEASE ----- */
DROP TABLE IF EXISTS release        CASCADE;
DROP TABLE IF EXISTS publisher      CASCADE;

/* ----- PHYSICAL BOOK ----- */
DROP TABLE IF EXISTS physical_book  CASCADE;
DROP TABLE IF EXISTS damage         CASCADE;

/* ----- USER ----- */
DROP TABLE IF EXISTS user_info      CASCADE;
DROP TABLE IF EXISTS student        CASCADE;
DROP TABLE IF EXISTS admin          CASCADE;

/* ----- BORROW ----- */ 
DROP TABLE IF EXISTS borrow         CASCADE;
DROP TABLE IF EXISTS fine           CASCADE;
DROP TABLE IF EXISTS transaction    CASCADE;