"""Assign pre-upgrade data to one explicitly chosen browser workspace.

Dry run by default. Back up the database before using --apply. No existing
ownership is reassigned. Run only as an operator with database credentials.
"""
import argparse
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import store

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace-id', required=True, help='64-character ID from authenticated GET /api/workspace')
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
if not re.fullmatch('[a-f0-9]{64}', args.workspace_id):
    parser.error('Invalid workspace ID')
resources = {'doc': ('documents', 'id'), 'quiz': ('quizzes', 'id'), 'exam': ('exam_papers', 'id'),
             'bank': ('question_banks', 'id'), 'attempt': ('quiz_attempts', 'id'),
             'exam_attempt': ('exam_attempts', 'id'), 'card': ('srs_cards', 'id'), 'mixed': ('mixed_papers', 'id')}
with store.connection() as conn:
    for kind, (table, column) in resources.items():
        rows = conn.execute(f'SELECT {column} AS id FROM {table} WHERE {column} NOT IN (SELECT resource_id FROM resource_owners WHERE kind = ?)', (kind,)).fetchall()
        print(f'{kind}: {len(rows)} unclaimed records' + (' (assigned)' if args.apply else ' (dry run)'))
        if args.apply:
            for row in rows:
                conn.execute('INSERT OR IGNORE INTO resource_owners VALUES (?, ?, ?)', (kind, row['id'], args.workspace_id))
# Old share links deliberately remain unclaimed and must be recreated.
