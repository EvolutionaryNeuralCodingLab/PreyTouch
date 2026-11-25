import csv
import io
import math
from pathlib import Path

import config

def parse_trajectory_csv(csv_text: str) -> list:
    """Read one bug trajectory, preserving the CSV row order."""
    reader = csv.DictReader(
        io.StringIO(csv_text.lstrip('\ufeff'), newline=''),
        strict=True
    )
    rows = []

    try:
        columns = reader.fieldnames or []
        expected = config.CUSTOM_BUGS_CSV_COLUMNS
        if len(columns) != len(expected) or set(columns) != set(expected):
            raise ValueError(f'CSV columns must be: {", ".join(expected)}')

        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'CSV line {line}: wrong number of values')

            for column in ('size', 'img-src'):
                value = row[column].strip()
                row[column] = None if value.lower() in ('', 'null') else value

            for column in ('x', 'y', 'angle', 'size'):
                if column == 'size' and row[column] is None:
                    continue
                try:
                    row[column] = float(row[column])
                except ValueError:
                    raise ValueError(
                        f'CSV line {line}: {column} must be a number'
                    ) from None
                if not math.isfinite(row[column]):
                    raise ValueError(
                        f'CSV line {line}: {column} must be finite'
                    )

            if row['size'] is not None and row['size'] < 0:
                raise ValueError(f'CSV line {line}: size cannot be negative')

            for column in ('strike-ability', 'is-reward'):
                value = row[column].strip()
                if value not in ('0', '1'):
                    raise ValueError(
                        f'CSV line {line}: {column} must be 0 or 1'
                    )
                row[column] = value == '1'

            rows.append(row)
    except csv.Error as exc:
        raise ValueError(f'Invalid CSV: {exc}') from exc

    if not rows:
        raise ValueError('CSV must contain at least one trajectory row')

    return rows

def copy_trajectories_to_block(trajectories: list, block_path: str) -> list:
    if not trajectories:
        raise ValueError('Custom bugs requires at least one trajectory CSV')

    upload_dir = (
        Path(config.EXPERIMENTS_DIR) / 'custom-bugs-inputs'
    ).resolve()
    output_dir = (Path(block_path) / 'trajectories').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for index, trajectory in enumerate(trajectories, start=1):
        source = Path(trajectory['path']).resolve()
        if source.parent != upload_dir or source.suffix.lower() != '.csv':
            raise ValueError('Trajectory must refer to an uploaded CSV')

        content = source.read_bytes()
        rows = parse_trajectory_csv(content.decode('utf-8-sig'))
        destination = output_dir / f'trajectory-{index:03d}.csv'
        with destination.open('xb') as saved_file:
            saved_file.write(content)

        copied.append({
            'name': trajectory['name'],
            'path': str(destination),
            'row_count': len(rows)
        })

    return copied