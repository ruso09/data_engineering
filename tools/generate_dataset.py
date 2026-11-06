"""
generate_dataset.py
====================
Generates a realistic, internally-consistent SYNTHETIC Indian e-commerce
dataset for Kaggle: customers, products, orders, order_items, payments,
shipments, returns, customer_reviews, marketing_campaigns.

Author: Synthetic Data Architect (generated with Claude)
License of the generated data: CC0 / open (see README.md)

Run:
    python3 generate_dataset.py

Outputs CSVs into ./data/
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
import datetime as dt
import math
import os

# --------------------------------------------------------------------------
# 0. CONFIG
# --------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("en_IN")
Faker.seed(SEED)

N_CUSTOMERS = 25_000
N_PRODUCTS = 550
N_ORDERS_TARGET = 100_000
START_DATE = dt.date(2023, 1, 1)
END_DATE = dt.date(2025, 12, 31)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUT_DIR, exist_ok=True)

DATE_RANGE = pd.date_range(START_DATE, END_DATE, freq="D")
N_DAYS = len(DATE_RANGE)

print(f"Date range: {START_DATE} to {END_DATE} ({N_DAYS} days)")

# --------------------------------------------------------------------------
# 1. REFERENCE / LOOKUP DATA
# --------------------------------------------------------------------------

STATE_CITY_MAP = {
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
    "Karnataka": ["Bengaluru", "Mysuru"],
    "Delhi": ["New Delhi"],
    "Rajasthan": ["Jaipur", "Jodhpur"],
    "Tamil Nadu": ["Chennai", "Coimbatore"],
    "Telangana": ["Hyderabad", "Warangal"],
    "West Bengal": ["Kolkata", "Howrah"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida"],
    "Kerala": ["Kochi", "Thiruvananthapuram"],
    "Punjab": ["Ludhiana", "Amritsar"],
    "Haryana": ["Gurugram", "Faridabad"],
    "Madhya Pradesh": ["Indore", "Bhopal"],
    "Bihar": ["Patna"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada"],
}
# state -> relative economic weight (affects customer counts / AOV skew)
STATE_WEIGHT = {
    "Gujarat": 1.0, "Maharashtra": 1.4, "Karnataka": 1.2, "Delhi": 1.3,
    "Rajasthan": 0.7, "Tamil Nadu": 1.1, "Telangana": 1.0, "West Bengal": 0.9,
    "Uttar Pradesh": 1.1, "Kerala": 0.8, "Punjab": 0.7, "Haryana": 0.9,
    "Madhya Pradesh": 0.7, "Bihar": 0.5, "Andhra Pradesh": 0.7,
}
CITY_LIST = [(c, s) for s, cities in STATE_CITY_MAP.items() for c in cities]
CITY_WEIGHTS = np.array([STATE_WEIGHT[s] for _, s in CITY_LIST], dtype=float)
CITY_WEIGHTS = CITY_WEIGHTS / CITY_WEIGHTS.sum()

WAREHOUSE_CITIES = ["Mumbai", "Bengaluru", "Delhi-NCR (Gurugram)", "Ahmedabad",
                    "Kolkata", "Hyderabad", "Chennai", "Lucknow"]

CATEGORIES = {
    # category: (min_price, max_price, margin_frac, return_rate_baseline, weight)
    "Electronics":            (3000, 80000, 0.12, 0.08, 9),
    "Mobile Accessories":     (150, 3000, 0.30, 0.06, 8),
    "Fashion":                (300, 5000, 0.40, 0.15, 12),
    "Beauty":                 (150, 3000, 0.35, 0.05, 6),
    "Home & Kitchen":         (200, 15000, 0.25, 0.07, 8),
    "Grocery":                (50, 2000, 0.15, 0.01, 10),
    "Sports":                 (300, 10000, 0.28, 0.06, 4),
    "Books":                  (100, 1500, 0.20, 0.02, 4),
    "Toys":                   (200, 4000, 0.30, 0.08, 3),
    "Furniture":              (2000, 50000, 0.20, 0.10, 3),
    "Footwear":               (400, 6000, 0.35, 0.18, 7),
    "Bags":                   (300, 5000, 0.35, 0.10, 4),
    "Watches":                (500, 20000, 0.30, 0.09, 3),
    "Personal Care":          (100, 2000, 0.30, 0.03, 5),
    "Appliances":             (1500, 60000, 0.15, 0.09, 4),
    "Stationery":             (50, 1000, 0.25, 0.02, 3),
    "Pet Supplies":           (150, 3000, 0.25, 0.04, 2),
    "Automotive Accessories": (200, 8000, 0.25, 0.07, 2),
    "Health & Wellness":      (150, 3000, 0.28, 0.03, 3),
    "Baby Care":              (150, 3000, 0.25, 0.05, 2),
    "Musical Instruments":    (500, 30000, 0.20, 0.08, 1),
    "Office Supplies":        (100, 5000, 0.22, 0.03, 2),
}
CAT_NAMES = list(CATEGORIES.keys())
CAT_WEIGHTS = np.array([v[4] for v in CATEGORIES.values()], dtype=float)
CAT_WEIGHTS = CAT_WEIGHTS / CAT_WEIGHTS.sum()

SUBCATEGORY_POOL = {
    "Electronics": ["Smartphones", "Laptops", "Headphones", "Smartwatches", "Cameras", "Tablets", "Speakers"],
    "Mobile Accessories": ["Chargers", "Cables", "Phone Cases", "Power Banks", "Screen Guards"],
    "Fashion": ["Men's Topwear", "Women's Ethnic Wear", "Men's Bottomwear", "Women's Western Wear", "Kids Wear", "Winter Wear"],
    "Beauty": ["Skincare", "Makeup", "Haircare", "Fragrances"],
    "Home & Kitchen": ["Cookware", "Dinnerware", "Storage & Containers", "Home Decor", "Bedding"],
    "Grocery": ["Staples", "Snacks", "Beverages", "Dairy", "Packaged Food"],
    "Sports": ["Fitness Equipment", "Outdoor Sports", "Yoga", "Cricket Gear", "Cycling"],
    "Books": ["Fiction", "Non-Fiction", "Academic", "Children's Books", "Comics"],
    "Toys": ["Educational Toys", "Action Figures", "Board Games", "Soft Toys"],
    "Furniture": ["Living Room", "Bedroom", "Office Furniture", "Storage"],
    "Footwear": ["Men's Casual", "Women's Casual", "Sports Shoes", "Sandals & Flip Flops", "Formal Shoes"],
    "Bags": ["Backpacks", "Handbags", "Travel Bags", "Wallets"],
    "Watches": ["Analog Watches", "Smart Watches", "Digital Watches"],
    "Personal Care": ["Oral Care", "Bath & Body", "Grooming"],
    "Appliances": ["Kitchen Appliances", "Large Appliances", "Air Conditioners", "Fans & Coolers"],
    "Stationery": ["Notebooks", "Pens & Pencils", "Art Supplies", "Office Stationery"],
    "Pet Supplies": ["Pet Food", "Pet Accessories", "Pet Grooming"],
    "Automotive Accessories": ["Car Accessories", "Bike Accessories", "Car Care"],
    "Health & Wellness": ["Supplements", "Ayurveda", "Medical Devices"],
    "Baby Care": ["Diapers", "Baby Food", "Baby Gear"],
    "Musical Instruments": ["String Instruments", "Percussion", "Electronic Instruments"],
    "Office Supplies": ["Printers", "Office Equipment", "Filing & Storage"],
}

BRAND_POOL = [fake.company().split(" ")[0] for _ in range(160)]
BRAND_POOL = list(dict.fromkeys(BRAND_POOL))  # dedupe, keep order

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Net Banking",
                    "Cash on Delivery", "Wallet", "EMI", "Pay Later",
                    "Gift Card", "Cardless EMI"]
PAYMENT_WEIGHTS = np.array([32, 15, 13, 8, 17, 7, 4, 2, 1, 1], dtype=float)
PAYMENT_WEIGHTS = PAYMENT_WEIGHTS / PAYMENT_WEIGHTS.sum()

MARKETING_CHANNELS = ["Organic Search", "Paid Search", "Social Media - Instagram",
                      "Social Media - Facebook", "Email Marketing", "Affiliate",
                      "Influencer Marketing", "Direct/App", "Referral", "SMS Marketing"]
CHANNEL_WEIGHTS = np.array([22, 14, 12, 9, 8, 7, 6, 12, 6, 4], dtype=float)
CHANNEL_WEIGHTS = CHANNEL_WEIGHTS / CHANNEL_WEIGHTS.sum()
# relative "customer quality" multiplier by acquisition channel (affects AOV/frequency)
CHANNEL_QUALITY = {
    "Organic Search": 1.05, "Paid Search": 0.95, "Social Media - Instagram": 0.9,
    "Social Media - Facebook": 0.85, "Email Marketing": 1.1, "Affiliate": 0.9,
    "Influencer Marketing": 0.95, "Direct/App": 1.2, "Referral": 1.15, "SMS Marketing": 0.85,
}

SHIPPING_METHODS = ["Standard", "Express", "Same-Day", "Next-Day", "Economy/Surface"]
SHIP_WEIGHTS = np.array([45, 20, 5, 15, 15], dtype=float)
SHIP_WEIGHTS = SHIP_WEIGHTS / SHIP_WEIGHTS.sum()
# base delivery days range (min,max) per shipping method (same-region)
SHIP_SPEED = {
    "Standard": (3, 6), "Express": (1, 3), "Same-Day": (0, 1),
    "Next-Day": (1, 2), "Economy/Surface": (5, 10),
}

CUSTOMER_SEGMENTS = ["Premium", "Regular", "Budget", "New"]
SEGMENT_WEIGHTS = np.array([12, 45, 30, 13], dtype=float)
SEGMENT_WEIGHTS = SEGMENT_WEIGHTS / SEGMENT_WEIGHTS.sum()
# lambda for poisson order-count generation, and AOV multiplier
SEGMENT_ORDER_LAMBDA = {"Premium": 9.0, "Regular": 4.2, "Budget": 2.0, "New": 1.0}
SEGMENT_SPEND_MULT = {"Premium": 1.8, "Regular": 1.0, "Budget": 0.6, "New": 0.8}

LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]

RETURN_REASONS = ["Product damaged", "Wrong product", "Size issue", "Quality issue",
                   "Changed mind", "Late delivery", "Product not as expected", "Other"]

COUPON_CODES = [None, None, None, None, "WELCOME10", "FEST20", "SAVE15", "BIGBILLION",
                "FLAT50", "NEW100", "APP5", "FREESHIP"]

print("Reference data ready.")

# --------------------------------------------------------------------------
# 2. SEASONALITY WEIGHT CURVE (daily demand multiplier)
# --------------------------------------------------------------------------

def festival_bumps(dates):
    """Return dict date->bump multiplier around major Indian shopping events."""
    bumps = {}
    festival_days = []
    for year in [2023, 2024, 2025]:
        # Republic Day
        festival_days.append((dt.date(year, 1, 26), 1.15, 1))
        # Holi (approx mid-March)
        holi = {2023: dt.date(2023, 3, 8), 2024: dt.date(2024, 3, 25), 2025: dt.date(2025, 3, 14)}[year]
        festival_days.append((holi, 1.25, 2))
        # Independence Day
        festival_days.append((dt.date(year, 8, 15), 1.15, 1))
        # Navratri/Dussehra (approx Oct)
        dussehra = {2023: dt.date(2023, 10, 24), 2024: dt.date(2024, 10, 12), 2025: dt.date(2025, 10, 2)}[year]
        festival_days.append((dussehra, 1.3, 4))
        # Diwali (biggest sale season)
        diwali = {2023: dt.date(2023, 11, 12), 2024: dt.date(2024, 11, 1), 2025: dt.date(2025, 10, 21)}[year]
        festival_days.append((diwali, 1.9, 6))
        # Eid al-Fitr (approx)
        eid = {2023: dt.date(2023, 4, 22), 2024: dt.date(2024, 4, 11), 2025: dt.date(2025, 3, 31)}[year]
        festival_days.append((eid, 1.2, 2))
        # Christmas / New Year / Year-End Sale
        festival_days.append((dt.date(year, 12, 25), 1.3, 5))
        festival_days.append((dt.date(year, 12, 31), 1.35, 3))
        # Republic-day-adjacent Republic sale, Summer sale (June), Independence sale
        festival_days.append((dt.date(year, 6, 15), 1.15, 3))

    for center, peak_mult, spread_days in festival_days:
        for offset in range(-spread_days, spread_days + 1):
            d = center + dt.timedelta(days=offset)
            if START_DATE <= d <= END_DATE:
                decay = max(0.0, 1 - abs(offset) / (spread_days + 1))
                mult = 1 + (peak_mult - 1) * decay
                bumps[d] = max(bumps.get(d, 1.0), mult)
    return bumps


def build_daily_weights():
    fb = festival_bumps(DATE_RANGE)
    weights = []
    for i, d in enumerate(DATE_RANGE):
        d_date = d.date()
        # base weekly pattern: weekend boost
        dow = d.weekday()  # 0=Mon
        weekly_mult = 1.25 if dow in (5, 6) else 1.0
        # monthly seasonality: slight Q4 boost, slight summer dip
        month = d_date.month
        month_mult = {1: 0.95, 2: 0.92, 3: 1.0, 4: 0.95, 5: 0.95, 6: 0.95,
                      7: 0.95, 8: 1.0, 9: 1.05, 10: 1.2, 11: 1.3, 12: 1.15}[month]
        # year-over-year growth (business growing over time)
        year = d_date.year
        yoy_mult = {2023: 0.85, 2024: 1.0, 2025: 1.2}[year]
        fest_mult = fb.get(d_date, 1.0)
        w = weekly_mult * month_mult * yoy_mult * fest_mult
        weights.append(w)
    weights = np.array(weights)
    return weights / weights.sum()


DAILY_WEIGHTS = build_daily_weights()
print("Seasonality curve built. Peak day weight:", DAILY_WEIGHTS.max(), "on",
      DATE_RANGE[DAILY_WEIGHTS.argmax()].date())

# --------------------------------------------------------------------------
# 3. CUSTOMERS
# --------------------------------------------------------------------------
print("Generating customers...")

customer_ids = np.array([f"CUST{100000 + i}" for i in range(N_CUSTOMERS)])

# signup dates: also seasonal-ish (more signups during festival/growth periods),
# but allow some slack before START_DATE isn't allowed (dataset starts 2023-01-01)
signup_day_idx = np.random.choice(N_DAYS, size=N_CUSTOMERS, p=DAILY_WEIGHTS)
# push a good chunk of signups earlier (base user pool existed since day 1)
early_mask = np.random.rand(N_CUSTOMERS) < 0.25
signup_day_idx[early_mask] = np.random.randint(0, 200, size=early_mask.sum())
signup_dates = DATE_RANGE[signup_day_idx]

genders = np.random.choice(["Male", "Female", "Other"], size=N_CUSTOMERS, p=[0.54, 0.44, 0.02])
ages = np.clip(np.random.normal(32, 9, N_CUSTOMERS).astype(int), 18, 70)


def age_group(a):
    if a < 25:
        return "18-24"
    elif a < 35:
        return "25-34"
    elif a < 45:
        return "35-44"
    elif a < 55:
        return "45-54"
    else:
        return "55+"


age_groups = np.array([age_group(a) for a in ages])

city_idx = np.random.choice(len(CITY_LIST), size=N_CUSTOMERS, p=CITY_WEIGHTS)
cust_cities = np.array([CITY_LIST[i][0] for i in city_idx])
cust_states = np.array([CITY_LIST[i][1] for i in city_idx])
pincode_prefix = np.random.randint(100, 900, size=N_CUSTOMERS)

segments = np.random.choice(CUSTOMER_SEGMENTS, size=N_CUSTOMERS, p=SEGMENT_WEIGHTS)
devices = np.random.choice(["Mobile App", "Mobile Web", "Desktop"], size=N_CUSTOMERS, p=[0.62, 0.23, 0.15])
preferred_payment = np.random.choice(PAYMENT_METHODS, size=N_CUSTOMERS, p=PAYMENT_WEIGHTS)
acquisition_channel = np.random.choice(MARKETING_CHANNELS, size=N_CUSTOMERS, p=CHANNEL_WEIGHTS)

customers = pd.DataFrame({
    "customer_id": customer_ids,
    "customer_signup_date": signup_dates.date,
    "gender": genders,
    "age": ages,
    "age_group": age_groups,
    "state": cust_states,
    "city": cust_cities,
    "pincode_prefix": pincode_prefix,
    "customer_segment": segments,
    "preferred_device": devices,
    "preferred_payment_method": preferred_payment,
    "acquisition_channel": acquisition_channel,
})

# each customer gets a hidden "activity span" (days active after signup) -> drives churn
max_possible_span = np.array([(END_DATE - d).days for d in signup_dates.date])
activity_span = np.minimum(
    max_possible_span,
    np.random.exponential(scale=500, size=N_CUSTOMERS).astype(int) + 30
)
customers["_activity_span_days"] = activity_span

print(f"Customers generated: {len(customers)}")

# --------------------------------------------------------------------------
# 4. PRODUCTS
# --------------------------------------------------------------------------
print("Generating products...")

product_ids = np.array([f"PROD{10000 + i}" for i in range(N_PRODUCTS)])
prod_cat_idx = np.random.choice(len(CAT_NAMES), size=N_PRODUCTS, p=CAT_WEIGHTS)
prod_categories = np.array([CAT_NAMES[i] for i in prod_cat_idx])

prod_subcats = []
prod_prices = []
prod_costs = []
prod_brands = []
prod_names = []
prod_return_baseline = []

ADJECTIVES = ["Premium", "Classic", "Pro", "Essential", "Deluxe", "Smart", "Ultra",
              "Everyday", "Compact", "Signature", "Advanced", "Basic"]

for cat in prod_categories:
    min_p, max_p, margin, ret_rate, _ = CATEGORIES[cat]
    sub = random.choice(SUBCATEGORY_POOL[cat])
    prod_subcats.append(sub)
    # log-uniform price within category range for realistic skew
    price = round(math.exp(np.random.uniform(math.log(min_p), math.log(max_p))), -1)
    price = max(price, min_p)
    prod_prices.append(price)
    cost = round(price * (1 - margin) * np.random.uniform(0.9, 1.05), 2)
    prod_costs.append(min(cost, price * 0.97))
    brand = random.choice(BRAND_POOL)
    prod_brands.append(brand)
    adj = random.choice(ADJECTIVES)
    prod_names.append(f"{brand} {adj} {sub[:-1] if sub.endswith('s') else sub}")
    # return rate baseline with small random noise per product
    prod_return_baseline.append(round(np.clip(np.random.normal(ret_rate, ret_rate * 0.2), 0.005, 0.5), 4))

rating_avg = np.clip(np.random.normal(4.0, 0.45, N_PRODUCTS), 1.5, 5.0).round(1)
rating_count = np.random.lognormal(mean=4.0, sigma=1.3, size=N_PRODUCTS).astype(int)
stock_qty = np.random.randint(0, 3000, size=N_PRODUCTS)
launch_day_idx = np.random.randint(0, N_DAYS, size=N_PRODUCTS)
launch_dates = DATE_RANGE[launch_day_idx].date
product_type = np.random.choice(["Standard", "Premium", "Budget", "Exclusive"], size=N_PRODUCTS,
                                 p=[0.55, 0.2, 0.2, 0.05])
discount_range = [f"{lo}-{lo+15}%" for lo in np.random.choice([0, 5, 10, 15, 20, 25, 30], size=N_PRODUCTS)]

products = pd.DataFrame({
    "product_id": product_ids,
    "product_name": prod_names,
    "category": prod_categories,
    "subcategory": prod_subcats,
    "brand": prod_brands,
    "price": prod_prices,
    "cost_price": prod_costs,
    "discount_range": discount_range,
    "rating_average": rating_avg,
    "rating_count": rating_count,
    "stock_quantity": stock_qty,
    "product_launch_date": launch_dates,
    "product_type": product_type,
    "return_rate_baseline": prod_return_baseline,
})
products["price"] = products["price"].astype(float)

print(f"Products generated: {len(products)}")

# --------------------------------------------------------------------------
# 5. ORDERS  (assign order counts per customer, then sample dates)
# --------------------------------------------------------------------------
print("Generating orders (this includes per-customer order-count + date sampling)...")

seg_lambda = customers["customer_segment"].map(SEGMENT_ORDER_LAMBDA).values
channel_q = customers["acquisition_channel"].map(CHANNEL_QUALITY).values
lam = seg_lambda * channel_q
raw_counts = np.random.poisson(lam)
# scale so total ~= N_ORDERS_TARGET
scale_factor = N_ORDERS_TARGET / max(raw_counts.sum(), 1)
order_counts = np.round(raw_counts * scale_factor).astype(int)
order_counts = np.clip(order_counts, 0, 400)
diff = N_ORDERS_TARGET - order_counts.sum()
# adjust random customers up/down to hit exact target
if diff != 0:
    idxs = np.random.choice(N_CUSTOMERS, size=abs(diff), replace=True)
    for idx in idxs:
        if diff > 0:
            order_counts[idx] += 1
        elif order_counts[idx] > 0:
            order_counts[idx] -= 1
customers["_order_count_target"] = order_counts
print("Total orders to generate:", order_counts.sum())

# Build per-order customer index array
cust_row_idx_per_order = np.repeat(np.arange(N_CUSTOMERS), order_counts)
np.random.shuffle(cust_row_idx_per_order)
n_orders = len(cust_row_idx_per_order)

# For each order, sample a date within [signup_date, signup_date+activity_span] weighted by seasonality
signup_idx_arr = signup_day_idx  # day index of signup, aligned with customer rows
span_arr = activity_span

order_signup_idx = signup_idx_arr[cust_row_idx_per_order]
order_span = span_arr[cust_row_idx_per_order]
order_end_idx = np.minimum(order_signup_idx + order_span, N_DAYS - 1)

# vectorized weighted sampling within [start, end] window using cumulative-weight inverse transform
cumw = np.cumsum(DAILY_WEIGHTS)
cumw_start = np.where(order_signup_idx > 0, cumw[np.clip(order_signup_idx - 1, 0, N_DAYS - 1)], 0.0)
cumw_end = cumw[order_end_idx]
u = np.random.uniform(0, 1, size=n_orders) * (cumw_end - cumw_start) + cumw_start
u = np.clip(u, 0, cumw[-1] - 1e-12)
order_day_idx = np.searchsorted(cumw, u, side="left")
order_day_idx = np.clip(order_day_idx, order_signup_idx, order_end_idx)

order_dates = DATE_RANGE[order_day_idx]

# order time (skewed towards evening 6pm-11pm, and lunchtime)
hour_weights = np.array([1,1,1,1,1,1,2,3,4,5,6,7,8,7,6,6,7,8,10,12,11,9,6,3], dtype=float)
hour_weights /= hour_weights.sum()
order_hours = np.random.choice(24, size=n_orders, p=hour_weights)
order_minutes = np.random.randint(0, 60, size=n_orders)
order_times = [f"{h:02d}:{m:02d}:{np.random.randint(0,60):02d}" for h, m in zip(order_hours, order_minutes)]

order_ids = np.array([f"ORD{1000000 + i}" for i in range(n_orders)])
order_customer_ids = customer_ids[cust_row_idx_per_order]

# delivery location: mostly same as home city/state, occasionally different (gifting / travel)
same_loc_mask = np.random.rand(n_orders) < 0.93
order_cities = cust_cities[cust_row_idx_per_order].copy()
order_states = cust_states[cust_row_idx_per_order].copy()
diff_idx = np.where(~same_loc_mask)[0]
if len(diff_idx) > 0:
    rand_city_idx = np.random.choice(len(CITY_LIST), size=len(diff_idx), p=CITY_WEIGHTS)
    order_cities[diff_idx] = [CITY_LIST[i][0] for i in rand_city_idx]
    order_states[diff_idx] = [CITY_LIST[i][1] for i in rand_city_idx]

shipping_method = np.random.choice(SHIPPING_METHODS, size=n_orders, p=SHIP_WEIGHTS)

# marketing channel for order: mostly matches customer acquisition channel, sometimes different (later exposure)
order_channel = acquisition_channel[cust_row_idx_per_order].copy()
switch_mask = np.random.rand(n_orders) < 0.25
order_channel[switch_mask] = np.random.choice(MARKETING_CHANNELS, size=switch_mask.sum(), p=CHANNEL_WEIGHTS)

# coupon / discount
coupon = np.random.choice(COUPON_CODES, size=n_orders, p=None)
has_coupon = np.array([c is not None for c in coupon])
discount_pct = np.where(has_coupon, np.random.choice([5, 10, 15, 20, 25, 30], size=n_orders), 0).astype(float)
# festival days get extra discount even w/o explicit coupon sometimes
fb_lookup = festival_bumps(DATE_RANGE)
is_festival_day = np.array([fb_lookup.get(d.date(), 1.0) > 1.1 for d in order_dates])
extra_disc_mask = is_festival_day & (~has_coupon) & (np.random.rand(n_orders) < 0.4)
discount_pct[extra_disc_mask] = np.random.choice([10, 15, 20], size=extra_disc_mask.sum())

# order status distribution, time-aware: very recent orders more likely "Processing"
days_from_end = (END_DATE - order_dates.date).astype("timedelta64[D]") if hasattr(order_dates, 'date') else None
days_from_end = np.array([(END_DATE - d.date()).days for d in order_dates])
status_choices = np.empty(n_orders, dtype=object)
recent_mask = days_from_end <= 5
status_choices[recent_mask] = np.random.choice(
    ["Processing", "Delivered", "Cancelled"], size=recent_mask.sum(), p=[0.55, 0.35, 0.10])
status_choices[~recent_mask] = np.random.choice(
    ["Delivered", "Cancelled", "Returned", "Failed"], size=(~recent_mask).sum(), p=[0.82, 0.08, 0.07, 0.03])

orders = pd.DataFrame({
    "order_id": order_ids,
    "customer_id": order_customer_ids,
    "order_date": order_dates.date,
    "order_time": order_times,
    "order_status": status_choices,
    "shipping_method": shipping_method,
    "delivery_city": order_cities,
    "delivery_state": order_states,
    "coupon_code": coupon,
    "discount_percentage": discount_pct,
    "marketing_channel": order_channel,
})
orders["_cust_row_idx"] = cust_row_idx_per_order  # helper, dropped later

print(f"Orders generated: {len(orders)}")
print(orders["order_status"].value_counts(normalize=True).round(3))

# --------------------------------------------------------------------------
# 6. ORDER ITEMS
# --------------------------------------------------------------------------
print("Generating order items...")

# number of distinct items per order (1-5), premium/regular customers buy slightly more
seg_per_order = customers["customer_segment"].values[cust_row_idx_per_order]
n_items_lambda = np.where(seg_per_order == "Premium", 2.0,
                  np.where(seg_per_order == "Regular", 1.5,
                  np.where(seg_per_order == "Budget", 1.2, 1.1)))
n_items = np.clip(np.random.poisson(n_items_lambda) + 1, 1, 5)

total_items = int(n_items.sum())
print("Total order_items rows:", total_items)

# repeat order-level arrays for each item
item_order_id = np.repeat(order_ids, n_items)
item_order_date_idx = np.repeat(order_day_idx, n_items)
item_customer_row = np.repeat(cust_row_idx_per_order, n_items)
item_order_discount = np.repeat(discount_pct, n_items)

# pick a product per item, weighted by category popularity & product rating_count (proxy popularity)
prod_popularity = (products["rating_count"].values + 5) / (products["rating_count"].values + 5).sum()
item_product_idx = np.random.choice(N_PRODUCTS, size=total_items, p=prod_popularity)

item_product_id = product_ids[item_product_idx]
item_unit_price = products["price"].values[item_product_idx]
item_unit_cost = products["cost_price"].values[item_product_idx]
item_category = products["category"].values[item_product_idx]

item_quantity = np.clip(np.random.poisson(1.3, size=total_items) + 1, 1, 8)
# small extra per-item discount noise on top of order discount (product-level promo)
item_extra_disc = np.random.choice([0, 0, 0, 5, 10], size=total_items)
item_discount_pct = np.clip(item_order_discount + item_extra_disc * 0.3, 0, 60)

discounted_unit_price = np.round(item_unit_price * (1 - item_discount_pct / 100), 2)
item_revenue = np.round(discounted_unit_price * item_quantity, 2)
item_cost = np.round(item_unit_cost * item_quantity, 2)
item_profit = np.round(item_revenue - item_cost, 2)

order_item_ids = np.array([f"OI{10000000 + i}" for i in range(total_items)])

order_items = pd.DataFrame({
    "order_item_id": order_item_ids,
    "order_id": item_order_id,
    "product_id": item_product_id,
    "quantity": item_quantity,
    "unit_price": item_unit_price,
    "discount_percentage": item_discount_pct.round(2),
    "item_revenue": item_revenue,
    "item_cost": item_cost,
    "profit": item_profit,
})
order_items["_category"] = item_category
order_items["_order_row_pos"] = np.repeat(np.arange(n_orders), n_items)  # helper

print(f"Order items generated: {len(order_items)}")

# --------------------------------------------------------------------------
# 7. RECONCILE ORDER TOTALS FROM ORDER ITEMS
# --------------------------------------------------------------------------
print("Reconciling order totals with order items...")

items_agg = order_items.groupby("order_id", sort=False).agg(
    subtotal=("item_revenue", "sum"),
    n_line_items=("order_item_id", "count"),
).reindex(order_ids).fillna(0.0)

orders["subtotal"] = items_agg["subtotal"].values.round(2)
orders["n_line_items"] = items_agg["n_line_items"].values.astype(int)

# shipping fee: depends on shipping method & subtotal (free shipping above threshold)
base_ship_fee = {"Standard": 49, "Express": 99, "Same-Day": 149, "Next-Day": 79, "Economy/Surface": 29}
orders["shipping_fee"] = orders["shipping_method"].map(base_ship_fee).astype(float)
orders.loc[orders["subtotal"] >= 499, "shipping_fee"] = 0.0

# tax: 5% GST approx (simplified) on subtotal
orders["tax_amount"] = (orders["subtotal"] * 0.05).round(2)

orders["final_amount"] = (orders["subtotal"] + orders["shipping_fee"] + orders["tax_amount"]).round(2)

# campaign_id assigned later (after campaigns table is built) based on marketing_channel + order month
print("Order totals reconciled. Sample final_amount stats:")
print(orders["final_amount"].describe().round(2))

# --------------------------------------------------------------------------
# 8. PAYMENTS
# --------------------------------------------------------------------------
print("Generating payments...")

# payment method leans toward customer's preferred method, with some variation
pref_pay = customers["preferred_payment_method"].values[cust_row_idx_per_order]
use_pref_mask = np.random.rand(n_orders) < 0.7
random_pay = np.random.choice(PAYMENT_METHODS, size=n_orders, p=PAYMENT_WEIGHTS)
payment_method = np.where(use_pref_mask, pref_pay, random_pay)

payment_status = np.empty(n_orders, dtype=object)
status_vals = orders["order_status"].values
payment_status[status_vals == "Delivered"] = "Success"
payment_status[status_vals == "Processing"] = np.random.choice(
    ["Success", "Pending"], size=(status_vals == "Processing").sum(), p=[0.85, 0.15])
payment_status[status_vals == "Cancelled"] = np.random.choice(
    ["Refunded", "Failed"], size=(status_vals == "Cancelled").sum(), p=[0.6, 0.4])
payment_status[status_vals == "Returned"] = "Refunded"
payment_status[status_vals == "Failed"] = "Failed"

# COD orders that failed/cancelled before dispatch never actually "paid"
is_cod = payment_method == "Cash on Delivery"
payment_status[is_cod & (status_vals == "Cancelled")] = "Not Charged"
payment_status[is_cod & (status_vals == "Failed")] = "Not Charged"

amount_paid = orders["final_amount"].values.copy()
amount_paid[np.isin(payment_status, ["Failed", "Not Charged", "Pending"])] = 0.0

transaction_fee = np.where(np.isin(payment_method, ["Credit Card", "EMI", "Cardless EMI"]),
                            (amount_paid * 0.015).round(2),
                            np.where(payment_method == "Wallet", (amount_paid * 0.005).round(2), 0.0))

refund_amount = np.zeros(n_orders)
refund_mask = payment_status == "Refunded"
refund_amount[refund_mask] = orders.loc[refund_mask, "final_amount"].values

# payment date: order date + 0-2 days for non-COD, or delivery-time for COD (approx order_date)
pay_day_offset = np.where(is_cod, 0, np.random.randint(0, 2, size=n_orders))
payment_day_idx = np.clip(order_day_idx + pay_day_offset, 0, N_DAYS - 1)
payment_dates = DATE_RANGE[payment_day_idx].date

payments = pd.DataFrame({
    "payment_id": [f"PAY{1000000+i}" for i in range(n_orders)],
    "order_id": order_ids,
    "payment_date": payment_dates,
    "payment_method": payment_method,
    "payment_status": payment_status,
    "amount_paid": amount_paid.round(2),
    "transaction_fee": transaction_fee,
    "refund_amount": refund_amount.round(2),
})
# refund can never exceed amount actually paid (use final_amount as basis, capped)
payments["refund_amount"] = np.minimum(payments["refund_amount"], orders["final_amount"].values)

print(f"Payments generated: {len(payments)}")

# --------------------------------------------------------------------------
# 9. SHIPMENTS  (only for orders that reached fulfillment: Delivered/Returned/Cancelled-after-dispatch)
# --------------------------------------------------------------------------
print("Generating shipments...")

has_shipment_mask = np.isin(status_vals, ["Delivered", "Returned", "Cancelled"])
# a portion of cancellations happen pre-dispatch (no shipment)
cancel_predispatch = (status_vals == "Cancelled") & (np.random.rand(n_orders) < 0.7)
has_shipment_mask = has_shipment_mask & (~cancel_predispatch)

ship_idx = np.where(has_shipment_mask)[0]
n_ship = len(ship_idx)

ship_order_id = order_ids[ship_idx]
ship_method = shipping_method[ship_idx]
ship_dest_city = order_cities[ship_idx]
ship_dest_state = order_states[ship_idx]
ship_order_day_idx = order_day_idx[ship_idx]

# assign warehouse: prefer nearest metro; simplified via state proximity groups
STATE_TO_WAREHOUSE = {
    "Gujarat": "Ahmedabad", "Maharashtra": "Mumbai", "Karnataka": "Bengaluru",
    "Delhi": "Delhi-NCR (Gurugram)", "Rajasthan": "Ahmedabad", "Tamil Nadu": "Chennai",
    "Telangana": "Hyderabad", "West Bengal": "Kolkata", "Uttar Pradesh": "Delhi-NCR (Gurugram)",
    "Kerala": "Chennai", "Punjab": "Delhi-NCR (Gurugram)", "Haryana": "Delhi-NCR (Gurugram)",
    "Madhya Pradesh": "Mumbai", "Bihar": "Kolkata", "Andhra Pradesh": "Hyderabad",
}
warehouse_city = np.array([STATE_TO_WAREHOUSE[s] for s in ship_dest_state])
same_region_mask = np.array([
    (STATE_TO_WAREHOUSE.get(ds, "") == wc) for ds, wc in zip(ship_dest_state, warehouse_city)
])

dispatch_offset = np.random.randint(0, 2, size=n_ship)  # dispatched within 0-1 day of order
dispatch_day_idx = np.clip(ship_order_day_idx + dispatch_offset, 0, N_DAYS - 1)

delivery_days = np.zeros(n_ship, dtype=int)
for method in SHIPPING_METHODS:
    m_mask = ship_method == method
    lo, hi = SHIP_SPEED[method]
    base_days = np.random.randint(lo, hi + 1, size=m_mask.sum())
    delivery_days[m_mask] = base_days
# cross-region deliveries take 1-3 extra days
delivery_days = delivery_days + np.where(~same_region_mask, np.random.randint(1, 4, size=n_ship), 0)
# random operational delay noise (small chance of extra delay)
extra_delay_mask = np.random.rand(n_ship) < 0.12
delivery_days = delivery_days + np.where(extra_delay_mask, np.random.randint(1, 5, size=n_ship), 0)
delivery_days = np.clip(delivery_days, 0, 20)

expected_delivery_day_idx = np.clip(dispatch_day_idx + np.round(
    np.vectorize(lambda m: (SHIP_SPEED[m][0] + SHIP_SPEED[m][1]) / 2)(ship_method)).astype(int), 0, N_DAYS - 1)
actual_delivery_day_idx = np.clip(dispatch_day_idx + delivery_days, 0, N_DAYS - 1)

ship_status_vals = status_vals[ship_idx]
delivery_status = np.where(ship_status_vals == "Returned", "Delivered then Returned",
                    np.where(ship_status_vals == "Cancelled", "Cancelled in Transit", "Delivered"))

delayed_flag = (actual_delivery_day_idx > expected_delivery_day_idx).astype(int)
# cancelled-in-transit orders have no real actual delivery date
actual_delivery_day_idx_final = actual_delivery_day_idx.copy().astype(float)
actual_delivery_day_idx_final[ship_status_vals == "Cancelled"] = np.nan
delayed_flag[ship_status_vals == "Cancelled"] = 0

shipments = pd.DataFrame({
    "shipment_id": [f"SHIP{1000000+i}" for i in range(n_ship)],
    "order_id": ship_order_id,
    "warehouse_city": warehouse_city,
    "delivery_city": ship_dest_city,
    "shipping_method": ship_method,
    "dispatch_date": DATE_RANGE[dispatch_day_idx].date,
    "expected_delivery_date": DATE_RANGE[expected_delivery_day_idx].date,
    "actual_delivery_date": [DATE_RANGE[int(i)].date() if not np.isnan(i) else pd.NaT
                              for i in actual_delivery_day_idx_final],
    "delivery_days": np.where(np.isnan(actual_delivery_day_idx_final), np.nan, delivery_days),
    "delivery_status": delivery_status,
    "delayed_flag": delayed_flag,
})

print(f"Shipments generated: {len(shipments)}")

# --------------------------------------------------------------------------
# 10. RETURNS
# --------------------------------------------------------------------------
print("Generating returns...")

# candidate items are those belonging to orders with status "Returned"
returned_order_set = set(order_ids[status_vals == "Returned"])
oi_is_returned_order = order_items["order_id"].isin(returned_order_set).values

# base return probability per item = category baseline * (much higher if order flagged Returned),
# scaled so that category-level return-rate differences remain clearly visible in the aggregate
cat_return_rate = order_items["_category"].map({c: v[3] for c, v in CATEGORIES.items()}).values
prob_if_returned_order = np.clip(cat_return_rate * 5.0, 0.2, 0.95)
prob_if_normal_order = cat_return_rate * 0.15
item_return_prob = np.where(oi_is_returned_order, prob_if_returned_order, prob_if_normal_order)
return_roll = np.random.rand(len(order_items)) < item_return_prob
# ensure every "Returned"-status order has at least one returned item
ret_df = order_items.loc[return_roll].copy()
missing_orders = returned_order_set - set(ret_df["order_id"])
if missing_orders:
    fallback = order_items[order_items["order_id"].isin(missing_orders)].drop_duplicates("order_id")
    ret_df = pd.concat([ret_df, fallback], ignore_index=True).drop_duplicates("order_item_id")

n_returns = len(ret_df)
ret_df = ret_df.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")

return_reason = np.random.choice(RETURN_REASONS, size=n_returns,
                                  p=[0.16, 0.12, 0.18, 0.14, 0.16, 0.08, 0.12, 0.04])
# return date: a few days after order date (approximate using order_day_idx via order_id map)
order_day_map = dict(zip(order_ids, order_day_idx))
ret_order_days = ret_df["order_id"].map(order_day_map).values
return_offset = np.random.randint(2, 15, size=n_returns)
return_day_idx = np.clip(ret_order_days + return_offset, 0, N_DAYS - 1)

return_status = np.random.choice(["Approved", "Rejected", "Pending", "Refund Completed"],
                                  size=n_returns, p=[0.15, 0.08, 0.07, 0.70])
refund_amt = np.where(np.isin(return_status, ["Approved", "Refund Completed"]),
                       ret_df["item_revenue"].values, 0.0)

returns = pd.DataFrame({
    "return_id": [f"RET{100000+i}" for i in range(n_returns)],
    "order_id": ret_df["order_id"].values,
    "customer_id": ret_df["customer_id"].values,
    "product_id": ret_df["product_id"].values,
    "return_date": DATE_RANGE[return_day_idx].date,
    "return_reason": return_reason,
    "refund_amount": refund_amt.round(2),
    "return_status": return_status,
})

print(f"Returns generated: {len(returns)}")

# --------------------------------------------------------------------------
# 11. CUSTOMER REVIEWS
# --------------------------------------------------------------------------
print("Generating customer reviews...")

# reviews sampled from delivered/returned orders' items (not all items get reviewed)
delivered_ret_order_set = set(order_ids[np.isin(status_vals, ["Delivered", "Returned"])])
oi_eligible = order_items["order_id"].isin(delivered_ret_order_set).values
review_prob = 0.35
review_roll = (np.random.rand(len(order_items)) < review_prob) & oi_eligible
rev_df = order_items.loc[review_roll].copy()
rev_df = rev_df.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
n_reviews = len(rev_df)

# delayed_flag per order for rating influence
ship_delay_map = dict(zip(shipments["order_id"], shipments["delayed_flag"]))
rev_delay = rev_df["order_id"].map(ship_delay_map).fillna(0).values
rev_returned = rev_df["order_id"].isin(returned_order_set).values
prod_rating_map = dict(zip(products["product_id"], products["rating_average"]))
rev_prod_rating = rev_df["product_id"].map(prod_rating_map).values

# base rating centered on product's average rating, penalized by delay/return
base = rev_prod_rating - 0.6 * rev_delay - 1.0 * rev_returned
noise = np.random.normal(0, 0.7, n_reviews)
rating = np.clip(np.round(base + noise), 1, 5).astype(int)

sentiment = np.where(rating >= 4, "Positive", np.where(rating == 3, "Neutral", "Negative"))

order_day_map2 = order_day_map
rev_order_days = rev_df["order_id"].map(order_day_map2).values
review_offset = np.random.randint(3, 30, size=n_reviews)
review_day_idx = np.clip(rev_order_days + review_offset, 0, N_DAYS - 1)

verified = np.random.rand(n_reviews) < 0.97

customer_reviews = pd.DataFrame({
    "review_id": [f"REV{100000+i}" for i in range(n_reviews)],
    "order_id": rev_df["order_id"].values,
    "customer_id": rev_df["customer_id"].values,
    "product_id": rev_df["product_id"].values,
    "review_date": DATE_RANGE[review_day_idx].date,
    "rating": rating,
    "review_sentiment": sentiment,
    "verified_purchase": verified,
})

print(f"Customer reviews generated: {len(customer_reviews)}")

# --------------------------------------------------------------------------
# 12. MARKETING CAMPAIGNS
# --------------------------------------------------------------------------
print("Generating marketing campaigns...")

CAMPAIGN_TYPES = ["Seasonal Sale", "Flash Sale", "New Product Launch", "Brand Awareness",
                  "Retargeting", "Loyalty Program", "Clearance Sale", "Festival Special"]

campaigns_rows = []
campaign_id_counter = 1
for year in [2023, 2024, 2025]:
    for month in range(1, 13):
        # 1-3 campaigns per month per a subset of channels
        active_channels = np.random.choice(MARKETING_CHANNELS,
                                           size=np.random.randint(3, 6), replace=False)
        for ch in active_channels:
            start = dt.date(year, month, np.random.randint(1, 10))
            duration = np.random.randint(5, 21)
            end = min(start + dt.timedelta(days=duration), END_DATE)
            if start > END_DATE:
                continue
            campaigns_rows.append({
                "campaign_id": f"CAMP{1000+campaign_id_counter}",
                "campaign_name": f"{ch} {random.choice(CAMPAIGN_TYPES)} {start.strftime('%b %Y')}",
                "channel": ch,
                "campaign_start_date": start,
                "campaign_end_date": end,
                "campaign_type": random.choice(CAMPAIGN_TYPES),
                "target_segment": random.choice(CUSTOMER_SEGMENTS + ["All"]),
                "discount_percentage": random.choice([5, 10, 15, 20, 25, 30]),
            })
            campaign_id_counter += 1

marketing_campaigns = pd.DataFrame(campaigns_rows)

# assign campaign_id to orders that had a coupon / were placed during an active campaign on same channel
orders_dates_arr = pd.to_datetime(orders["order_date"])
camp_by_channel = {ch: marketing_campaigns[marketing_campaigns["channel"] == ch] for ch in MARKETING_CHANNELS}


def assign_campaign(row_channel, row_date):
    df = camp_by_channel.get(row_channel)
    if df is None or df.empty:
        return None
    match = df[(pd.to_datetime(df["campaign_start_date"]) <= row_date) &
               (pd.to_datetime(df["campaign_end_date"]) >= row_date)]
    if len(match) == 0:
        return None
    return match.iloc[0]["campaign_id"]


# vectorize via merge_asof-like approach per channel for performance
orders["campaign_id"] = None
orders["_odate_ts"] = orders_dates_arr
for ch in MARKETING_CHANNELS:
    ch_mask = orders["marketing_channel"] == ch
    if not ch_mask.any():
        continue
    cdf = camp_by_channel[ch].copy()
    if cdf.empty:
        continue
    cdf["campaign_start_date"] = pd.to_datetime(cdf["campaign_start_date"])
    cdf["campaign_end_date"] = pd.to_datetime(cdf["campaign_end_date"])
    sub = orders.loc[ch_mask, ["_odate_ts"]].copy()
    # for each order date, find campaign whose window contains it (use interval index)
    intervals = pd.IntervalIndex.from_arrays(cdf["campaign_start_date"], cdf["campaign_end_date"], closed="both")
    campaign_ids_for_dates = []
    for d in sub["_odate_ts"]:
        pos = intervals.get_indexer([d])[0]
        campaign_ids_for_dates.append(cdf.iloc[pos]["campaign_id"] if pos != -1 else None)
    orders.loc[ch_mask, "campaign_id"] = campaign_ids_for_dates

orders.drop(columns=["_odate_ts"], inplace=True)

# now compute campaign performance (impressions/clicks/conversions/cost/revenue) from actual order data
camp_order_counts = orders.groupby("campaign_id").agg(
    conversions=("order_id", "count"),
    revenue_generated=("final_amount", "sum"),
).reset_index()

marketing_campaigns = marketing_campaigns.merge(camp_order_counts, on="campaign_id", how="left")
marketing_campaigns["conversions"] = marketing_campaigns["conversions"].fillna(0).astype(int)
marketing_campaigns["revenue_generated"] = marketing_campaigns["revenue_generated"].fillna(0.0).round(2)

# channel-specific conversion-rate assumptions to back-calculate realistic clicks/impressions
CHANNEL_CVR = {  # conversion rate: conversions / clicks
    "Organic Search": 0.09, "Paid Search": 0.05, "Social Media - Instagram": 0.03,
    "Social Media - Facebook": 0.025, "Email Marketing": 0.12, "Affiliate": 0.06,
    "Influencer Marketing": 0.04, "Direct/App": 0.15, "Referral": 0.18, "SMS Marketing": 0.07,
}
CHANNEL_CTR = {  # click-through rate: clicks / impressions
    "Organic Search": 0.20, "Paid Search": 0.04, "Social Media - Instagram": 0.02,
    "Social Media - Facebook": 0.018, "Email Marketing": 0.15, "Affiliate": 0.06,
    "Influencer Marketing": 0.03, "Direct/App": 0.30, "Referral": 0.25, "SMS Marketing": 0.10,
}
# Funnel metrics (impressions/clicks) are illustrative and derived from conversions using
# channel-specific CTR/CVR assumptions.
conv = marketing_campaigns["conversions"].values.astype(float)
cvr = marketing_campaigns["channel"].map(CHANNEL_CVR).values
ctr = marketing_campaigns["channel"].map(CHANNEL_CTR).values

clicks = np.round(conv / np.maximum(cvr, 0.01) * np.random.uniform(0.9, 1.1, len(conv)))
impressions = np.round(clicks / np.maximum(ctr, 0.005) * np.random.uniform(0.9, 1.1, len(conv)))

# Campaign spend is calibrated directly as a realistic fraction of attributed revenue
# (marketing spend typically runs 8-45% of the revenue it drives), so ROI comes out in a
# believable range instead of an artifact of an unrelated CPC/CVR chain.
CHANNEL_ROI_TARGET = {  # (mean, std) of target ROI multiplier per channel
    "Organic Search": (4.5, 1.5), "Direct/App": (5.0, 1.8), "Referral": (4.0, 1.5),
    "Email Marketing": (3.2, 1.2), "SMS Marketing": (1.8, 1.0),
    "Affiliate": (1.5, 0.8), "Paid Search": (1.3, 0.9),
    "Social Media - Instagram": (0.9, 0.9), "Social Media - Facebook": (0.7, 0.9),
    "Influencer Marketing": (0.5, 1.0),
}
revenue = marketing_campaigns["revenue_generated"].values
target_roi = np.array([
    np.random.normal(*CHANNEL_ROI_TARGET[ch]) for ch in marketing_campaigns["channel"].values
])
target_roi = np.clip(target_roi, -0.6, 12.0)  # cap extreme outliers, allow some loss-making campaigns

campaign_cost = np.where(revenue > 0, revenue / (1 + target_roi), np.random.uniform(5000, 20000, len(revenue)))
campaign_cost = np.maximum(campaign_cost, 3000).round(2)

marketing_campaigns["impressions"] = impressions.astype(int)
marketing_campaigns["clicks"] = clicks.astype(int)
marketing_campaigns["campaign_cost"] = campaign_cost
marketing_campaigns["roi"] = ((marketing_campaigns["revenue_generated"] - marketing_campaigns["campaign_cost"])
                              / marketing_campaigns["campaign_cost"]).round(3)

print(f"Marketing campaigns generated: {len(marketing_campaigns)}")

# --------------------------------------------------------------------------
# 13. RECOMPUTE DERIVED CUSTOMER STATISTICS FROM ACTUAL ORDER HISTORY
# --------------------------------------------------------------------------
print("Recomputing derived customer statistics from order history...")

# Only count orders that materially count as "purchases" (exclude Failed)
counted_orders = orders[orders["order_status"] != "Failed"]
cust_stats = counted_orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    total_spend=("final_amount", "sum"),
    last_order_date=("order_date", "max"),
).reindex(customer_ids)

customers = customers.merge(cust_stats, left_on="customer_id", right_index=True, how="left")
customers["total_orders"] = customers["total_orders"].fillna(0).astype(int)
customers["total_spend"] = customers["total_spend"].fillna(0.0).round(2)
customers["average_order_value"] = np.where(
    customers["total_orders"] > 0,
    (customers["total_spend"] / customers["total_orders"]).round(2),
    0.0,
)

# customer_status derived from recency relative to dataset end date
REFERENCE_DATE = END_DATE
days_since_last_order = customers["last_order_date"].apply(
    lambda d: (REFERENCE_DATE - d).days if pd.notnull(d) else np.nan)
customers["_days_since_last_order"] = days_since_last_order


def derive_status(row):
    if row["total_orders"] == 0:
        return "Never Purchased"
    d = row["_days_since_last_order"]
    if d <= 90:
        return "Active"
    elif d <= 180:
        return "At Risk"
    else:
        return "Churned"


customers["customer_status"] = customers.apply(derive_status, axis=1)

# loyalty tier derived from total_spend quartiles among purchasing customers
purchasers = customers[customers["total_orders"] > 0]
if len(purchasers) > 0:
    q = purchasers["total_spend"].quantile([0.5, 0.8, 0.95]).values

    def tier(spend, orders_n):
        if orders_n == 0:
            return "Bronze"
        if spend >= q[2]:
            return "Platinum"
        elif spend >= q[1]:
            return "Gold"
        elif spend >= q[0]:
            return "Silver"
        return "Bronze"

    customers["loyalty_tier"] = customers.apply(lambda r: tier(r["total_spend"], r["total_orders"]), axis=1)
else:
    customers["loyalty_tier"] = "Bronze"

customers.drop(columns=["_activity_span_days", "_order_count_target", "_days_since_last_order"],
              inplace=True, errors="ignore")

print("Customer stats recomputed.")
print(customers["customer_status"].value_counts())
print(customers["loyalty_tier"].value_counts())

# clean up helper columns on orders / order_items
orders.drop(columns=["_cust_row_idx", "n_line_items"], inplace=True, errors="ignore")
order_items.drop(columns=["_category", "_order_row_pos"], inplace=True, errors="ignore")

print("\nAll tables generated successfully.")

# --------------------------------------------------------------------------
# 14. FINAL DATA-QUALITY GUARDS
# --------------------------------------------------------------------------
print("Applying final data-quality guards...")

# no negative prices/quantities
assert (order_items["quantity"] > 0).all()
assert (order_items["unit_price"] >= 0).all()
assert (products["price"] > 0).all()

# refund_amount must never exceed amount_paid (payments) — enforce
payments["refund_amount"] = np.minimum(payments["refund_amount"], payments["amount_paid"].where(
    payments["amount_paid"] > 0, orders.set_index("order_id").loc[payments["order_id"], "final_amount"].values))
payments["refund_amount"] = payments["refund_amount"].clip(lower=0).round(2)

# returns refund_amount must never exceed the item's revenue (already true by construction) - re-clip for safety
returns["refund_amount"] = returns["refund_amount"].clip(lower=0).round(2)

# introduce a small, realistic amount of MCAR missing data in a few non-critical columns
rng = np.random.default_rng(SEED)


def sprinkle_missing(df, col, frac):
    idx = df.sample(frac=frac, random_state=SEED).index
    df.loc[idx, col] = np.nan


sprinkle_missing(customers, "age", 0.01)
sprinkle_missing(customers, "preferred_device", 0.005)
sprinkle_missing(products, "rating_average", 0.008)
sprinkle_missing(orders, "coupon_code", 0.0)  # already mostly None by design, skip extra
sprinkle_missing(customer_reviews, "review_sentiment", 0.003)

print("Data-quality guards applied.")

# --------------------------------------------------------------------------
# 15. SAVE CSV FILES
# --------------------------------------------------------------------------
print("Saving CSV files...")

customers.to_csv(os.path.join(OUT_DIR, "customers.csv"), index=False)
products.to_csv(os.path.join(OUT_DIR, "products.csv"), index=False)
orders.to_csv(os.path.join(OUT_DIR, "orders.csv"), index=False)
order_items.to_csv(os.path.join(OUT_DIR, "order_items.csv"), index=False)
payments.to_csv(os.path.join(OUT_DIR, "payments.csv"), index=False)
shipments.to_csv(os.path.join(OUT_DIR, "shipments.csv"), index=False)
returns.to_csv(os.path.join(OUT_DIR, "returns.csv"), index=False)
customer_reviews.to_csv(os.path.join(OUT_DIR, "customer_reviews.csv"), index=False)
marketing_campaigns.to_csv(os.path.join(OUT_DIR, "marketing_campaigns.csv"), index=False)

print("All CSV files saved to:", OUT_DIR)

# --------------------------------------------------------------------------
# 16. VALIDATION REPORT
# --------------------------------------------------------------------------
print("Running validation checks...")

checks = []


def add_check(name, expected, actual, passed, notes=""):
    checks.append({
        "validation_check": name,
        "expected_condition": expected,
        "actual_result": actual,
        "status": "PASS" if passed else "FAIL",
        "notes": notes,
    })


# Row counts vs requirements
add_check("customers row count", ">= 25000", len(customers), len(customers) >= 25000)
add_check("products row count", ">= 500", len(products), len(products) >= 500)
add_check("orders row count", ">= 100000", len(orders), len(orders) >= 100000)
add_check("unique states", ">= 15", customers["state"].nunique(), customers["state"].nunique() >= 15)
add_check("unique cities", ">= 25", customers["city"].nunique(), customers["city"].nunique() >= 25)
add_check("unique categories", ">= 20", products["category"].nunique(), products["category"].nunique() >= 20)
add_check("payment methods", ">= 10", payments["payment_method"].nunique(), payments["payment_method"].nunique() >= 10)
add_check("marketing channels", ">= 8", orders["marketing_channel"].nunique(), orders["marketing_channel"].nunique() >= 8)
add_check("shipping methods", ">= 5", orders["shipping_method"].nunique(), orders["shipping_method"].nunique() >= 5)

# Uniqueness of primary keys
for name, df, col in [
    ("customers", customers, "customer_id"), ("products", products, "product_id"),
    ("orders", orders, "order_id"), ("order_items", order_items, "order_item_id"),
    ("payments", payments, "payment_id"), ("shipments", shipments, "shipment_id"),
    ("returns", returns, "return_id"), ("customer_reviews", customer_reviews, "review_id"),
    ("marketing_campaigns", marketing_campaigns, "campaign_id"),
]:
    dup = df[col].duplicated().sum()
    add_check(f"{name}.{col} uniqueness", "0 duplicates", int(dup), dup == 0)

# Foreign key integrity
fk_checks = [
    ("orders.customer_id -> customers", orders["customer_id"], customers["customer_id"]),
    ("order_items.order_id -> orders", order_items["order_id"], orders["order_id"]),
    ("order_items.product_id -> products", order_items["product_id"], products["product_id"]),
    ("payments.order_id -> orders", payments["order_id"], orders["order_id"]),
    ("shipments.order_id -> orders", shipments["order_id"], orders["order_id"]),
    ("returns.order_id -> orders", returns["order_id"], orders["order_id"]),
    ("returns.customer_id -> customers", returns["customer_id"], customers["customer_id"]),
    ("returns.product_id -> products", returns["product_id"], products["product_id"]),
    ("customer_reviews.order_id -> orders", customer_reviews["order_id"], orders["order_id"]),
    ("customer_reviews.customer_id -> customers", customer_reviews["customer_id"], customers["customer_id"]),
    ("customer_reviews.product_id -> products", customer_reviews["product_id"], products["product_id"]),
]
for name, child, parent in fk_checks:
    orphans = (~child.isin(parent)).sum()
    add_check(f"FK: {name}", "0 orphan rows", int(orphans), orphans == 0)

# Date range validity
for name, s in [("customers.customer_signup_date", customers["customer_signup_date"]),
               ("orders.order_date", orders["order_date"]),
               ("products.product_launch_date", products["product_launch_date"])]:
    s_dt = pd.to_datetime(s)
    within = ((s_dt.dt.date >= START_DATE) & (s_dt.dt.date <= END_DATE)).all()
    add_check(f"date range: {name}", f"{START_DATE} to {END_DATE}",
              f"{s_dt.min().date()} to {s_dt.max().date()}", within)

# Numerical range checks
add_check("order_items.quantity > 0", "all > 0", int((order_items['quantity'] <= 0).sum()),
          (order_items['quantity'] > 0).all(), "count of violations shown as actual_result if any")
add_check("products.price > 0", "all > 0", int((products['price'] <= 0).sum()), (products['price'] > 0).all())
add_check("orders.final_amount >= 0", "all >= 0", int((orders['final_amount'] < 0).sum()),
          (orders['final_amount'] >= 0).all())

# Order total reconciliation: subtotal must equal sum of item_revenue per order
recon = order_items.groupby("order_id")["item_revenue"].sum().round(2)
orders_sub = orders.set_index("order_id")["subtotal"].round(2)
mismatch = (recon.reindex(orders_sub.index).fillna(0) - orders_sub).abs()
n_mismatch = (mismatch > 0.5).sum()
add_check("order subtotal reconciles with order_items", "0 mismatches (tolerance 0.5)",
          int(n_mismatch), n_mismatch == 0)

# Payment reconciliation: refund_amount <= amount that could have been paid (final_amount)
pay_merged = payments.merge(orders[["order_id", "final_amount"]], on="order_id")
bad_refund = (pay_merged["refund_amount"] > pay_merged["final_amount"] + 0.5).sum()
add_check("payments.refund_amount <= final_amount", "0 violations", int(bad_refund), bad_refund == 0)

# Return refund <= item revenue
ret_merged = returns.merge(order_items[["order_item_id", "order_id", "product_id", "item_revenue"]],
                            left_on=["order_id", "product_id"], right_on=["order_id", "product_id"], how="left")
ret_check = ret_merged.groupby(["order_id", "product_id"]).agg(
    refund=("refund_amount", "first"), rev=("item_revenue", "max")).reset_index()
bad_ret = (ret_check["refund"] > ret_check["rev"] + 0.5).sum()
add_check("returns.refund_amount <= item_revenue", "0 violations", int(bad_ret), bad_ret == 0)

# Shipment date consistency: actual_delivery_date >= dispatch_date (where present)
ship_valid = shipments.dropna(subset=["actual_delivery_date"])
bad_dates = (pd.to_datetime(ship_valid["actual_delivery_date"]) < pd.to_datetime(ship_valid["dispatch_date"])).sum()
add_check("shipments.actual_delivery_date >= dispatch_date", "0 violations", int(bad_dates), bad_dates == 0)

# expected_delivery_date >= dispatch_date
bad_exp = (pd.to_datetime(shipments["expected_delivery_date"]) < pd.to_datetime(shipments["dispatch_date"])).sum()
add_check("shipments.expected_delivery_date >= dispatch_date", "0 violations", int(bad_exp), bad_exp == 0)

# Customer aggregate consistency: recompute total_spend independently & compare
recheck = counted_orders.groupby("customer_id")["final_amount"].sum().round(2)
cust_spend = customers.set_index("customer_id")["total_spend"]
diff2 = (recheck.reindex(cust_spend.index).fillna(0) - cust_spend.fillna(0)).abs()
n_diff2 = (diff2 > 0.5).sum()
add_check("customers.total_spend matches order history", "0 mismatches (tolerance 0.5)",
          int(n_diff2), n_diff2 == 0)

# City/state validity
valid_pairs = set((c, s) for c, s in CITY_LIST)
bad_pairs = (~orders[["delivery_city", "delivery_state"]].apply(tuple, axis=1).isin(valid_pairs)).sum()
add_check("orders city/state combinations valid", "0 invalid pairs", int(bad_pairs), bad_pairs == 0)

# Null percentage spot-check (should be low, non-zero only where intentionally sprinkled)
null_pct_age = customers["age"].isna().mean()
add_check("customers.age null percentage", "< 2%", f"{null_pct_age:.2%}", null_pct_age < 0.02)

validation_report = pd.DataFrame(checks)
validation_report.to_csv(os.path.join(OUT_DIR, "dataset_validation_report.csv"), index=False)

n_fail = (validation_report["status"] == "FAIL").sum()
print(f"Validation complete: {len(validation_report)} checks run, {n_fail} FAILED.")
if n_fail > 0:
    print(validation_report[validation_report["status"] == "FAIL"])

# --------------------------------------------------------------------------
# 17. DATA DICTIONARY
# --------------------------------------------------------------------------
print("Building data dictionary...")

dict_rows = []


def dd(table, col, dtype, desc, example, allowed, nullable, meaning):
    dict_rows.append({
        "table_name": table, "column_name": col, "data_type": dtype, "description": desc,
        "example": example, "allowed_values": allowed, "nullable": nullable, "business_meaning": meaning,
    })


# customers.csv
dd("customers", "customer_id", "string", "Unique customer identifier", "CUST100001", "CUSTxxxxxx", "No", "Primary key")
dd("customers", "customer_signup_date", "date", "Date the customer registered", "2023-04-12", "2023-01-01 to 2025-12-31", "No", "Acquisition date")
dd("customers", "gender", "string", "Self-reported gender", "Female", "Male/Female/Other", "No", "Demographic")
dd("customers", "age", "int", "Customer age in years", "29", "18-70", "Yes (~1%)", "Demographic")
dd("customers", "age_group", "string", "Binned age group", "25-34", "18-24/25-34/35-44/45-54/55+", "No", "Demographic segment")
dd("customers", "state", "string", "Indian state of residence", "Gujarat", "15 Indian states", "No", "Geography")
dd("customers", "city", "string", "Indian city of residence", "Ahmedabad", "31 Indian cities", "No", "Geography")
dd("customers", "pincode_prefix", "int", "First 3 digits of postal PIN code (synthetic)", "382", "100-899", "No", "Geography proxy")
dd("customers", "customer_segment", "string", "Latent value segment used to drive behavior", "Premium", "Premium/Regular/Budget/New", "No", "Marketing segmentation")
dd("customers", "preferred_device", "string", "Most-used shopping device", "Mobile App", "Mobile App/Mobile Web/Desktop", "Yes (~0.5%)", "UX analytics")
dd("customers", "preferred_payment_method", "string", "Customer's usual payment method", "UPI", "10 payment methods", "No", "Payments analytics")
dd("customers", "acquisition_channel", "string", "Marketing channel that acquired the customer", "Organic Search", "10 channels", "No", "Marketing attribution")
dd("customers", "total_orders", "int", "Count of non-Failed orders placed (derived)", "12", ">= 0", "No", "RFM - Frequency")
dd("customers", "total_spend", "float", "Sum of final_amount across non-Failed orders (derived)", "45210.50", ">= 0", "No", "RFM - Monetary / CLV base")
dd("customers", "average_order_value", "float", "total_spend / total_orders (derived)", "3767.54", ">= 0", "No", "Spending behavior")
dd("customers", "last_order_date", "date", "Date of most recent non-Failed order (derived)", "2025-11-02", "date or null", "Yes (never purchased)", "RFM - Recency")
dd("customers", "customer_status", "string", "Derived from recency of last_order_date vs 2025-12-31: Active <=90d, At Risk <=180d, Churned >180d, Never Purchased if 0 orders", "Active", "Active/At Risk/Churned/Never Purchased", "No", "Churn label")
dd("customers", "loyalty_tier", "string", "Derived from total_spend quartile among purchasers", "Gold", "Bronze/Silver/Gold/Platinum", "No", "Loyalty program tier")

# products.csv
dd("products", "product_id", "string", "Unique product identifier", "PROD10001", "PRODxxxxx", "No", "Primary key")
dd("products", "product_name", "string", "Generated product title", "Nike Premium Sneaker", "free text", "No", "Catalog display")
dd("products", "category", "string", "Top-level product category", "Footwear", "22 categories", "No", "Catalog taxonomy")
dd("products", "subcategory", "string", "Product subcategory", "Sports Shoes", "varies by category", "No", "Catalog taxonomy")
dd("products", "brand", "string", "Synthetic brand name", "Nike", "~150 brands", "No", "Catalog attribute")
dd("products", "price", "float", "Listed selling price (MRP-like)", "2499.00", "> 0", "No", "Pricing")
dd("products", "cost_price", "float", "Internal cost price used for margin calc", "1499.40", "> 0, < price", "No", "Profitability")
dd("products", "discount_range", "string", "Typical discount band shown for this product", "10-25%", "text range", "No", "Merchandising")
dd("products", "rating_average", "float", "Average product star rating", "4.2", "1.0-5.0", "Yes (~0.8%)", "Product quality signal")
dd("products", "rating_count", "int", "Number of ratings received", "312", ">= 0", "No", "Popularity proxy")
dd("products", "stock_quantity", "int", "Units currently in stock (synthetic snapshot)", "540", ">= 0", "No", "Inventory")
dd("products", "product_launch_date", "date", "Date product was first listed", "2023-06-01", "2023-01-01 to 2025-12-31", "No", "Catalog lifecycle")
dd("products", "product_type", "string", "Merchandising tier", "Premium", "Standard/Premium/Budget/Exclusive", "No", "Merchandising")
dd("products", "return_rate_baseline", "float", "Underlying (hidden) probability driving return simulation for this product", "0.12", "0.005-0.5", "No", "Simulation parameter / leakage risk if used post-hoc")

# orders.csv
dd("orders", "order_id", "string", "Unique order identifier", "ORD1000001", "ORDxxxxxxx", "No", "Primary key")
dd("orders", "customer_id", "string", "Customer who placed the order", "CUST100001", "FK -> customers", "No", "Relationship")
dd("orders", "order_date", "date", "Date order was placed", "2024-11-01", "2023-01-01 to 2025-12-31", "No", "Transaction timestamp")
dd("orders", "order_time", "string", "Time of day order was placed (HH:MM:SS)", "20:14:03", "00:00:00-23:59:59", "No", "Behavioral analytics")
dd("orders", "order_status", "string", "Final status of the order", "Delivered", "Delivered/Cancelled/Returned/Failed/Processing", "No", "Order lifecycle / target variable")
dd("orders", "shipping_method", "string", "Selected shipping method", "Express", "5 methods", "No", "Fulfillment")
dd("orders", "delivery_city", "string", "Destination city", "Pune", "31 cities", "No", "Geography")
dd("orders", "delivery_state", "string", "Destination state", "Maharashtra", "15 states", "No", "Geography")
dd("orders", "coupon_code", "string", "Coupon applied, if any", "FEST20", "coupon list or null", "Yes (~65%)", "Promotions")
dd("orders", "discount_percentage", "float", "Order-level discount percentage applied", "15.0", "0-30", "No", "Pricing")
dd("orders", "marketing_channel", "string", "Channel attributed for this order", "Paid Search", "10 channels", "No", "Marketing attribution")
dd("orders", "subtotal", "float", "Sum of item_revenue across order_items for this order (derived)", "18500.00", ">= 0", "No", "Revenue base")
dd("orders", "shipping_fee", "float", "Shipping charge (waived above INR499 subtotal)", "49.00", ">= 0", "No", "Revenue component")
dd("orders", "tax_amount", "float", "Approximate 5% GST on subtotal", "925.00", ">= 0", "No", "Revenue component")
dd("orders", "final_amount", "float", "subtotal + shipping_fee + tax_amount (derived)", "19474.00", ">= 0", "No", "Order value / revenue target")
dd("orders", "campaign_id", "string", "Marketing campaign active for this channel/date, if any", "CAMP1042", "FK -> marketing_campaigns or null", "Yes", "Marketing attribution")

# order_items.csv
dd("order_items", "order_item_id", "string", "Unique line-item identifier", "OI10000001", "OIxxxxxxxx", "No", "Primary key")
dd("order_items", "order_id", "string", "Parent order", "ORD1000001", "FK -> orders", "No", "Relationship")
dd("order_items", "product_id", "string", "Product purchased", "PROD10001", "FK -> products", "No", "Relationship")
dd("order_items", "quantity", "int", "Units purchased of this product in this order", "2", "1-8", "No", "Volume")
dd("order_items", "unit_price", "float", "Product list price at time of purchase", "2499.00", "> 0", "No", "Pricing")
dd("order_items", "discount_percentage", "float", "Line-item discount percentage", "15.0", "0-60", "No", "Pricing")
dd("order_items", "item_revenue", "float", "quantity * discounted unit price (derived)", "4248.30", ">= 0", "No", "Revenue")
dd("order_items", "item_cost", "float", "quantity * cost_price (derived)", "2998.80", ">= 0", "No", "Cost")
dd("order_items", "profit", "float", "item_revenue - item_cost (derived)", "1249.50", "can be negative", "No", "Profitability target")

# payments.csv
dd("payments", "payment_id", "string", "Unique payment record identifier", "PAY1000001", "PAYxxxxxxx", "No", "Primary key")
dd("payments", "order_id", "string", "Order being paid for", "ORD1000001", "FK -> orders", "No", "Relationship")
dd("payments", "payment_date", "date", "Date payment was processed", "2024-11-01", "2023-01-01 to 2025-12-31", "No", "Transaction timestamp")
dd("payments", "payment_method", "string", "Method used", "UPI", "10 methods", "No", "Payments analytics")
dd("payments", "payment_status", "string", "Outcome of the payment", "Success", "Success/Pending/Failed/Refunded/Not Charged", "No", "Fraud/ops target")
dd("payments", "amount_paid", "float", "Amount actually captured", "19474.00", ">= 0", "No", "Cashflow")
dd("payments", "transaction_fee", "float", "Payment-gateway fee charged to merchant", "292.11", ">= 0", "No", "Cost")
dd("payments", "refund_amount", "float", "Amount refunded, if any (<= amount that could be paid)", "0.00", ">= 0", "No", "Cashflow")

# shipments.csv
dd("shipments", "shipment_id", "string", "Unique shipment identifier", "SHIP1000001", "SHIPxxxxxxx", "No", "Primary key")
dd("shipments", "order_id", "string", "Associated order", "ORD1000001", "FK -> orders", "No", "Relationship")
dd("shipments", "warehouse_city", "string", "Dispatching warehouse hub", "Mumbai", "8 hub cities", "No", "Fulfillment")
dd("shipments", "delivery_city", "string", "Destination city", "Pune", "31 cities", "No", "Geography")
dd("shipments", "shipping_method", "string", "Method used for this shipment", "Express", "5 methods", "No", "Fulfillment")
dd("shipments", "dispatch_date", "date", "Date parcel left warehouse", "2024-11-02", "2023-01-01 to 2025-12-31", "No", "Fulfillment - available pre-delivery")
dd("shipments", "expected_delivery_date", "date", "SLA-promised delivery date (available at order time)", "2024-11-04", "2023-01-01 to 2025-12-31", "No", "Feature for delay-prediction (safe, no leakage)")
dd("shipments", "actual_delivery_date", "date", "Date parcel was actually delivered", "2024-11-05", "date or null (cancelled-in-transit)", "Yes", "TARGET / leakage risk - do not use as an input feature to predict delayed_flag")
dd("shipments", "delivery_days", "float", "actual_delivery_date - dispatch_date, in days", "3", ">= 0 or null", "Yes", "Derived from actual date - leakage risk, same caution as actual_delivery_date")
dd("shipments", "delivery_status", "string", "Final shipment outcome", "Delivered", "Delivered/Delivered then Returned/Cancelled in Transit", "No", "Fulfillment outcome")
dd("shipments", "delayed_flag", "int (0/1)", "1 if actual_delivery_date > expected_delivery_date", "0", "0/1", "No", "TARGET variable for delay prediction")

# returns.csv
dd("returns", "return_id", "string", "Unique return record identifier", "RET100001", "RETxxxxxx", "No", "Primary key")
dd("returns", "order_id", "string", "Order the return belongs to", "ORD1000001", "FK -> orders", "No", "Relationship")
dd("returns", "customer_id", "string", "Customer who returned the item", "CUST100001", "FK -> customers", "No", "Relationship")
dd("returns", "product_id", "string", "Product returned", "PROD10001", "FK -> products", "No", "Relationship")
dd("returns", "return_date", "date", "Date return was initiated", "2024-11-10", "2023-01-01 to 2025-12-31", "No", "Transaction timestamp")
dd("returns", "return_reason", "string", "Stated reason for return", "Size issue", "8 reasons", "No", "Return-driver analysis")
dd("returns", "refund_amount", "float", "Amount refunded for this return (<= item_revenue)", "2499.00", ">= 0", "No", "Cashflow")
dd("returns", "return_status", "string", "Current status of the return", "Refund Completed", "Approved/Rejected/Pending/Refund Completed", "No", "Ops target")

# customer_reviews.csv
dd("customer_reviews", "review_id", "string", "Unique review identifier", "REV100001", "REVxxxxxx", "No", "Primary key")
dd("customer_reviews", "order_id", "string", "Order the review is linked to", "ORD1000001", "FK -> orders", "No", "Relationship")
dd("customer_reviews", "customer_id", "string", "Reviewer", "CUST100001", "FK -> customers", "No", "Relationship")
dd("customer_reviews", "product_id", "string", "Product reviewed", "PROD10001", "FK -> products", "No", "Relationship")
dd("customer_reviews", "review_date", "date", "Date review was posted", "2024-11-20", "2023-01-01 to 2025-12-31", "No", "Transaction timestamp")
dd("customer_reviews", "rating", "int", "Star rating given", "4", "1-5", "No", "Satisfaction target")
dd("customer_reviews", "review_sentiment", "string", "Sentiment label derived from rating", "Positive", "Positive/Neutral/Negative", "Yes (~0.3%)", "NLP / sentiment target")
dd("customer_reviews", "verified_purchase", "bool", "Whether review came from a verified purchase", "True", "True/False", "No", "Trust signal")

# marketing_campaigns.csv
dd("marketing_campaigns", "campaign_id", "string", "Unique campaign identifier", "CAMP1001", "CAMPxxxx", "No", "Primary key")
dd("marketing_campaigns", "campaign_name", "string", "Human-readable campaign name", "Paid Search Flash Sale Nov 2024", "free text", "No", "Marketing catalog")
dd("marketing_campaigns", "channel", "string", "Marketing channel used", "Paid Search", "10 channels", "No", "Attribution")
dd("marketing_campaigns", "campaign_start_date", "date", "Campaign start date", "2024-11-01", "2023-01-01 to 2025-12-31", "No", "Campaign window")
dd("marketing_campaigns", "campaign_end_date", "date", "Campaign end date", "2024-11-15", "2023-01-01 to 2025-12-31", "No", "Campaign window")
dd("marketing_campaigns", "campaign_type", "string", "Type of campaign", "Flash Sale", "8 types", "No", "Marketing catalog")
dd("marketing_campaigns", "target_segment", "string", "Intended customer segment", "Premium", "Premium/Regular/Budget/New/All", "No", "Targeting")
dd("marketing_campaigns", "discount_percentage", "float", "Headline discount offered", "20.0", "5-30", "No", "Promotions")
dd("marketing_campaigns", "impressions", "int", "Estimated ad impressions (derived from conversions & channel CTR/CVR assumptions)", "185000", ">= 0", "No", "Marketing funnel")
dd("marketing_campaigns", "clicks", "int", "Estimated clicks (derived)", "9200", ">= 0", "No", "Marketing funnel")
dd("marketing_campaigns", "conversions", "int", "Actual orders attributed to this campaign (from orders table)", "410", ">= 0", "No", "Marketing funnel - ground truth")
dd("marketing_campaigns", "campaign_cost", "float", "Estimated media spend for the campaign", "58400.00", ">= 0", "No", "Marketing cost")
dd("marketing_campaigns", "revenue_generated", "float", "Sum of final_amount for attributed orders (from orders table)", "512300.00", ">= 0", "No", "Marketing ROI - ground truth")
dd("marketing_campaigns", "roi", "float", "(revenue_generated - campaign_cost) / campaign_cost", "7.77", "can be negative", "No", "Marketing performance")

data_dictionary = pd.DataFrame(dict_rows)
data_dictionary.to_csv(os.path.join(OUT_DIR, "data_dictionary.csv"), index=False)
print(f"Data dictionary written with {len(data_dictionary)} rows.")

print("\n=== GENERATION COMPLETE ===")
print("Files written to:", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, f)
    print(f"  {f:35s} {os.path.getsize(fp)/1024:>10.1f} KB")


