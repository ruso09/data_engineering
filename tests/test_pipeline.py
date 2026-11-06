import csv
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from ecommerce.pipeline import build, convert

FIXTURE = Path(__file__).parent / 'fixtures'

class PipelineTests(unittest.TestCase):
    def test_replay_and_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'warehouse.sqlite'
            first = build(FIXTURE, target)
            with closing(sqlite3.connect(target)) as db:
                before = db.execute('SELECT * FROM mart_daily_sales ORDER BY order_date').fetchall()
                self.assertEqual(db.execute('SELECT COUNT(*) FROM fact_orders').fetchone()[0], first['tables']['orders']['rows'])
                self.assertAlmostEqual(db.execute('SELECT SUM(net_collected) FROM fact_orders').fetchone()[0], db.execute('SELECT SUM(amount_paid-refund_amount) FROM payments').fetchone()[0], places=5)
            second = build(FIXTURE, target)
            self.assertEqual(first['tables'], second['tables'])
            with closing(sqlite3.connect(target)) as db:
                self.assertEqual(before, db.execute('SELECT * FROM mart_daily_sales ORDER BY order_date').fetchall())

    def test_invalid_snapshot_preserves_target(self):
        for failure in ('duplicate', 'orphan', 'bad_total', 'missing_file'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / 'source'
                shutil.copytree(FIXTURE, source)
                target = Path(directory) / 'warehouse.sqlite'
                build(source, target)
                before = target.read_bytes()
                path = source / 'orders.csv'
                with path.open(newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    fields, rows = reader.fieldnames, list(reader)
                if failure == 'duplicate':
                    rows.append(rows[0])
                elif failure == 'orphan':
                    rows[0]['customer_id'] = 'UNKNOWN'
                elif failure == 'bad_total':
                    rows[0]['subtotal'] = '0'
                with path.open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                if failure == 'missing_file':
                    path.unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    build(source, target)
                self.assertEqual(before, target.read_bytes())

    def test_numeric_contract(self):
        spec = {'data_type': 'int', 'nullable': 'No', 'column_name': 'quantity'}
        for value in ('1.5', 'NaN', 'inf', ''):
            with self.assertRaises(ValueError):
                convert(value, spec)
        self.assertEqual(convert('4.0', spec), 4)

if __name__ == '__main__':
    unittest.main()
