"""Validated, atomic full-snapshot ecommerce warehouse. Python standard library only."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from datetime import date, datetime, timezone

KEYS = dict(customers='customer_id', products='product_id', orders='order_id',
            order_items='order_item_id', payments='payment_id', shipments='shipment_id',
            returns='return_id', customer_reviews='review_id', marketing_campaigns='campaign_id')
PARENTS = dict(customer_id='customers', product_id='products', order_id='orders', campaign_id='marketing_campaigns')

def ident(value):
    if not re.fullmatch(r'[a-z_]+', value):
        raise ValueError(f'Unsafe identifier: {value}')
    return '"' + value + '"'

def convert(value, spec):
    if value == '':
        if spec['nullable'].startswith('No'):
            raise ValueError(f"Required value missing: {spec['column_name']}")
        return None
    kind = spec['data_type']
    if kind in ('int', 'float'):
        number = float(value)
        if not math.isfinite(number) or (kind == 'int' and not number.is_integer()):
            raise ValueError(f'Invalid {kind}: {value}')
        return int(number) if kind == 'int' else number
    if kind == 'date':
        date.fromisoformat(value)
    return value

def build(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with (source / 'data_dictionary.csv').open(encoding='utf-8-sig', newline='') as f:
        dictionary = list(csv.DictReader(f))
    handle, temporary = tempfile.mkstemp(suffix='.sqlite', dir=target.parent)
    os.close(handle)
    db = sqlite3.connect(temporary)
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'tables': {}, 'checks': {}}
    try:
        for table, key in KEYS.items():
            specs = {s['column_name']: s for s in dictionary if s['table_name'] == table}
            path = source / (table + '.csv')
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(1048576), b''):
                    digest.update(block)
            with path.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames
                if not columns or len(set(columns)) != len(columns) or set(columns) != set(specs):
                    raise ValueError(f'{table}: schema differs from data dictionary')
                definitions = []
                for column in columns:
                    spec = specs[column]
                    dtype = {'int': 'INTEGER', 'float': 'REAL'}.get(spec['data_type'], 'TEXT')
                    constraint = ' PRIMARY KEY NOT NULL' if column == key else ''
                    definitions.append(ident(column) + ' ' + dtype + constraint)
                db.execute(f'CREATE TABLE {ident(table)} ({", ".join(definitions)})')
                insert = f'INSERT INTO {ident(table)} VALUES ({",".join("?" for _ in columns)})'
                count = 0
                for line, row in enumerate(reader, 2):
                    try:
                        if None in row or any(v is None for v in row.values()):
                            raise ValueError('Malformed CSV row')
                        db.execute(insert, [convert(row[c], specs[c]) for c in columns])
                    except (ValueError, sqlite3.Error) as exc:
                        raise ValueError(f'{table}, line {line}: {exc}') from exc
                    count += 1
                if not count:
                    raise ValueError(f'{table}: empty snapshot')
                report['tables'][table] = {'rows': count, 'sha256': digest.hexdigest()}
                for column in columns:
                    if column in PARENTS and column != key:
                        db.execute(f'CREATE INDEX {ident(table + "_" + column)} ON {ident(table)} ({ident(column)})')
        def check(name, sql):
            failures = db.execute(sql).fetchone()[0]
            report['checks'][name] = failures
            if failures:
                raise ValueError(f'{name}: {failures} invalid rows')
        for table, key in KEYS.items():
            columns = [r[1] for r in db.execute(f'PRAGMA table_info({ident(table)})')]
            for column in columns:
                if column in PARENTS and column != key:
                    parent = PARENTS[column]
                    check(f'{table}.{column}_foreign_key', f'SELECT COUNT(*) FROM {ident(table)} c LEFT JOIN {ident(parent)} p ON c.{ident(column)} = p.{ident(column)} WHERE c.{ident(column)} IS NOT NULL AND p.{ident(column)} IS NULL')
        check('payment_order_grain', 'SELECT COUNT(*) FROM (SELECT order_id FROM payments GROUP BY order_id HAVING COUNT(*) > 1)')
        check('shipment_order_grain', 'SELECT COUNT(*) FROM (SELECT order_id FROM shipments GROUP BY order_id HAVING COUNT(*) > 1)')
        check('order_line_reconciliation', '''SELECT COUNT(*) FROM orders o LEFT JOIN
            (SELECT order_id, SUM(item_revenue) revenue FROM order_items GROUP BY order_id) i USING(order_id)
            WHERE i.revenue IS NULL OR ABS(o.subtotal-i.revenue)>0.5''')
        check('order_total_reconciliation', 'SELECT COUNT(*) FROM orders WHERE ABS(final_amount-subtotal-shipping_fee-tax_amount)>0.5')
        check('positive_quantity', 'SELECT COUNT(*) FROM order_items WHERE quantity <= 0 OR item_revenue < 0 OR item_cost < 0')
        check('payment_refund_bounds', 'SELECT COUNT(*) FROM payments p JOIN orders o USING(order_id) WHERE p.refund_amount < 0 OR p.refund_amount > o.final_amount+0.5')
        check('shipment_dates', 'SELECT COUNT(*) FROM shipments WHERE actual_delivery_date < dispatch_date OR expected_delivery_date < dispatch_date')
        sql = Path(__file__).with_name('marts.sql').read_text(encoding='utf-8')
        db.executescript(sql)
        db.execute('CREATE TABLE pipeline_run (report_json TEXT NOT NULL)')
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        db.execute('INSERT INTO pipeline_run VALUES (?)', (json.dumps(report),))
        db.commit()
        db.close()
        os.replace(temporary, target)
        return report
    finally:
        db.close()
        if os.path.exists(temporary):
            os.unlink(temporary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, default=Path('output/ecommerce.sqlite'))
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.target), indent=2))
