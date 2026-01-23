import argparse
from .keyframe_handler import KeyframeHandler

def main():
    parser = argparse.ArgumentParser(description="Keyframe simplification CLI for CVAT jobs.")
    parser.add_argument("--job-id", type=int, required=True, help="CVAT job ID to process")
    parser.add_argument(
        "--field", action="append",
        help="Keyframe simplification field in the format: field,threshold,method. Example: --field position,5.1,l2 --field rotation,0.0,l_inf"
    )
    parser.add_argument("--undo", action="store_true", help="Restore annotations from the latest backup")

    args = parser.parse_args()
    handler = KeyframeHandler()

    if args.undo:
        print(f"Restoring job {args.job_id} from backup...")
        handler.restore_from_backup(args.job_id)
    else:
        if not args.field:
            parser.error("--field is required when not using --undo")
        fields = KeyframeHandler.parse_fields(args.field)
        print(f"Starting automatic processing for job_id={args.job_id}")
        handler.simplifying_job(args.job_id, fields)

if __name__ == "__main__":
    main()
