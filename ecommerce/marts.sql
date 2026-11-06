-- Grain: one order. Aggregate children before joining to prevent fan-out.
CREATE TABLE fact_orders AS
WITH items AS (
 SELECT order_id, SUM(quantity) units, SUM(item_revenue) merchandise_revenue,
 SUM(item_cost) merchandise_cost, SUM(profit) merchandise_profit
 FROM order_items GROUP BY order_id
), refunds AS (
 SELECT order_id, COUNT(*) return_records FROM returns GROUP BY order_id
)
SELECT o.*, i.units, i.merchandise_revenue, i.merchandise_cost, i.merchandise_profit,
 p.amount_paid, p.refund_amount payment_refund_amount,
 p.amount_paid-p.refund_amount net_collected,
 s.delayed_flag, COALESCE(r.return_records,0) return_records
FROM orders o JOIN items i USING(order_id)
LEFT JOIN payments p USING(order_id)
LEFT JOIN shipments s USING(order_id)
LEFT JOIN refunds r USING(order_id);
CREATE UNIQUE INDEX fact_orders_key ON fact_orders(order_id);

-- Gross ordered value includes every status; it is not recognized revenue.
CREATE TABLE mart_daily_sales AS
SELECT order_date, COUNT(*) orders, SUM(units) units,
 ROUND(SUM(final_amount),2) gross_ordered_value,
 ROUND(SUM(net_collected),2) net_collected,
 SUM(CASE WHEN order_status='Cancelled' THEN 1 ELSE 0 END) cancelled_orders
FROM fact_orders GROUP BY order_date;

CREATE TABLE mart_customer_value AS
SELECT c.customer_id, c.state, c.customer_segment, COUNT(f.order_id) orders,
 MAX(f.order_date) last_order_date, ROUND(COALESCE(SUM(f.net_collected),0),2) net_collected
FROM customers c LEFT JOIN fact_orders f USING(customer_id)
GROUP BY c.customer_id, c.state, c.customer_segment;

CREATE TABLE mart_product_sales AS
SELECT p.product_id, p.category, SUM(i.quantity) ordered_units,
 ROUND(SUM(i.item_revenue),2) ordered_merchandise_value,
 ROUND(SUM(i.profit),2) ordered_merchandise_profit
FROM products p JOIN order_items i USING(product_id)
GROUP BY p.product_id, p.category;
