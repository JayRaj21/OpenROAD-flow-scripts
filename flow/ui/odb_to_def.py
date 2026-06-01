"""
Export a DEF file from an ODB checkpoint.
Run via: openroad -python /work/ui/odb_to_def.py --odb <path> --out <path>
"""
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--odb', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()

try:
    import odb
    db = odb.dbDatabase.create()
    odb.read_db(db, args.odb)
    block = db.getChip().getBlock()
    odb.write_def(block, args.out)
    print(f"Wrote DEF to {args.out}")
except Exception as e:
    # Fallback: try via openroad module
    try:
        from openroad import Tech, Design
        tech = Tech()
        design = Design(tech)
        design.readDb(args.odb)
        design.writeDef(args.out)
        print(f"Wrote DEF to {args.out}")
    except Exception as e2:
        print(f"ERROR: {e2}", file=sys.stderr)
        sys.exit(1)
