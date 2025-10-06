import csv
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import backfill_watchdog_mae_mfe as backfill


LOG_HEADERS = [
    'closed_at',
    'product_id',
    'position_side',
    'net_size',
    'leverage',
    'opened_at',
    'closure_reason',
    'entry_price',
    'exit_price',
    'profit_loss',
    'profit_loss_pct',
    'mae',
    'mfe',
    'duration_seconds',
]


class BackfillWatchdogMaeMfeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.csv_path = Path(self.temp_dir.name) / 'watchdog_closed_positions.csv'

    def _write_rows(self, rows: list[Dict[str, str]]) -> None:
        with self.csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_HEADERS)
            writer.writeheader()
            writer.writerows(rows)

    def _read_rows(self) -> list[Dict[str, str]]:
        with self.csv_path.open('r', newline='') as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    def test_backfill_updates_missing_values(self) -> None:
        self._write_rows([
            {
                'closed_at': '2025-10-05T12:00:00Z',
                'product_id': 'BTC-PERP-INTX',
                'position_side': 'LONG',
                'net_size': '2',
                'leverage': '',
                'opened_at': '2025-10-05T06:00:00Z',
                'closure_reason': 'take_profit',
                'entry_price': '100',
                'exit_price': '102',
                'profit_loss': '4',
                'profit_loss_pct': '2',
                'mae': '',
                'mfe': '',
                'duration_seconds': '21600',
            }
        ])

        fake_cb = MagicMock()

        with patch.object(
            backfill,
            'compute_mae_mfe_from_history',
            return_value=(-12.34, 45.67),
        ) as mock_compute:
            stats = backfill.backfill_file(self.csv_path, cb=fake_cb, dry_run=False)

        self.assertEqual(stats.rows_updated, 1)
        self.assertEqual(stats.failures, 0)
        mock_compute.assert_called_once()

        rows = self._read_rows()
        self.assertEqual(rows[0]['mae'], '-12.34')
        self.assertEqual(rows[0]['mfe'], '45.67')

    def test_dry_run_does_not_modify_file(self) -> None:
        self._write_rows([
            {
                'closed_at': '2025-10-05T12:00:00Z',
                'product_id': 'ETH-PERP-INTX',
                'position_side': 'SHORT',
                'net_size': '-5',
                'leverage': '',
                'opened_at': '2025-10-05T08:00:00Z',
                'closure_reason': 'stop_loss',
                'entry_price': '2000',
                'exit_price': '2025',
                'profit_loss': '-125',
                'profit_loss_pct': '-0.62',
                'mae': '',
                'mfe': '',
                'duration_seconds': '14400',
            }
        ])

        with patch.object(
            backfill,
            'compute_mae_mfe_from_history',
            return_value=(-50.0, 80.0),
        ) as mock_compute:
            stats = backfill.backfill_file(self.csv_path, cb=MagicMock(), dry_run=True)

        self.assertEqual(stats.rows_updated, 1)
        mock_compute.assert_called_once()

        rows = self._read_rows()
        self.assertEqual(rows[0]['mae'], '')
        self.assertEqual(rows[0]['mfe'], '')

    def test_missing_entry_price_skips_row(self) -> None:
        self._write_rows([
            {
                'closed_at': '2025-10-05T12:00:00Z',
                'product_id': 'SOL-PERP-INTX',
                'position_side': 'LONG',
                'net_size': '10',
                'leverage': '',
                'opened_at': '2025-10-05T10:00:00Z',
                'closure_reason': 'expired',
                'entry_price': '',
                'exit_price': '25',
                'profit_loss': '0',
                'profit_loss_pct': '0',
                'mae': '',
                'mfe': '',
                'duration_seconds': '7200',
            }
        ])

        with patch.object(
            backfill,
            'compute_mae_mfe_from_history',
        ) as mock_compute:
            stats = backfill.backfill_file(self.csv_path, cb=MagicMock(), dry_run=False)

        mock_compute.assert_not_called()
        self.assertEqual(stats.rows_updated, 0)
        self.assertEqual(stats.rows_skipped, 1)


if __name__ == '__main__':
    unittest.main()
