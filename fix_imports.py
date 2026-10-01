import sys

def add_sys_path(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    if "import sys" not in content:
        insert_code = """
import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)
"""
        lines = content.split('\n')
        # Insert after the first import or docstring
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i
                break

        lines.insert(insert_idx, insert_code.strip())

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Added sys.path to {filepath}")

add_sys_path('credit-card-fraud-detection/src/train.py')
add_sys_path('credit-card-fraud-detection/src/data_preprocessing.py')
