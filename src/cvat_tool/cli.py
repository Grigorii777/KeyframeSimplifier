
import argparse
import dotenv
from .keyframe_handler import KeyframeHandler

def main():
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Keyframe simplification CLI for CVAT jobs.")
    parser.add_argument("--job-id", type=int, required=True, help="CVAT job ID to process")
    parser.add_argument(
        "--field", action="append",
        help="Keyframe simplification field in the format: field,threshold,method. Example: --field position,5.1,l2 --field rotation,0.0,l_inf. If not specified, auto-calculated based on --auto-percent."
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.9,
        help="IoU threshold for keyframe selection (0.0 to 1.0). Lower values = more simplification. Values > 1.0 disable IoU-based selection. Default: 0.9"
    )
    parser.add_argument(
        "--auto-percent", type=float, default=0,
        help="Percentage for auto-calculating thresholds when --field is not specified. Set to 0 to disable (default). Example: --auto-percent 1.0"
    )
    parser.add_argument("--undo", action="store_true", help="Restore annotations from the latest backup")
    parser.add_argument("--size-check", action="store_true", help="Check and display frames where object size differs from the first keyframe")

    args = parser.parse_args()
    handler = KeyframeHandler()

    if args.undo:
        print(f"Restoring job {args.job_id} from backup...")
        handler.restore_from_backup(args.job_id)
    elif args.size_check:
        print(f"Checking object sizes for job {args.job_id}...")
        handler.check_object_sizes(args.job_id)
    else:
        fields = None
        if args.field:
            fields = KeyframeHandler.parse_fields(args.field)
        print(f"Starting automatic processing for job_id={args.job_id}")
        handler.simplifying_job(args.job_id, fields, iou_threshold=args.iou_threshold, auto_percent=args.auto_percent)

if __name__ == "__main__":
    main()
