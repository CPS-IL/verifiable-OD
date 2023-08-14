
import json
import matplotlib.pyplot as plt
import os
import pprint

from compare import GT_ID_INDEX, get_argparser

COVERAGE_INDEX = 13


def get_args():
    global args
    parser = get_argparser()
    parser.add_argument('--infile', type=str, default="top_view_coverage.json",
                        help='Name of the input file')
    parser.add_argument('--gt_covered_ratio', type=float, default=0.75,
                        help="Ratio of GT that needs to be covered by detections to count as TP")
    parser.add_argument('--fn_outfile', type=str, default="top_view_coverage_FN.json",
                        help='Name of the output file')
    args = parser.parse_args()
    return args


def save_fn_list(args, infiles):
    total_fn_count = 0
    for filepath in infiles:
        for part in filepath.split("/"):
            if "segment" in part:
                segment_name = part
                break

        with open(filepath) as gt_file:
            gts = json.load(gt_file)
            fns = dict()

        for frame in gts.keys():
            fn = []
            for gt in gts[frame]:
                if gt[COVERAGE_INDEX] < args.gt_covered_ratio:
                    fn.append(gt)
            fns[frame] = fn
            if len(fn) > 0:
                total_fn_count += len(fn)
                print(segment_name, frame, fn[0][GT_ID_INDEX])
                pprint.pprint(fn)
        with open(filepath.replace(args.infile, args.fn_outfile), "w") as fn_file:
            json.dump(fns, fn_file, indent = 4)
    print("Total FN Count", total_fn_count)


def main(args):

    if args.segment:
        segment_paths = [os.path.realpath(args.segment)]
    else:
        segment_paths = ([os.path.join(args.dataset, f) for f in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset, f))])

    infiles = []
    for segment_path in segment_paths:
        infile = os.path.join(segment_path, "top_views", args.infile)
        if not os.path.exists(infile):
            print("[WARN] Input file does not exist", infile)
            continue
        infiles.append(infile)

    save_fn_list(args, infiles)


if __name__ == '__main__':
    main(get_args())
