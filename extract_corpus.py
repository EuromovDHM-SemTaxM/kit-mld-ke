import sys

from vtk.util._argparse import ArgumentParser

if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = ArgumentParser()

    if len(argv) == 0:
        parser.print_help()
        parser.exit(1)
    parser.add_argument("dataset_path", type=str,
                        help="Path to the directory containing the kit-mld dataset")
    parser.add_argument("--gold_format", "-g", choices=["original", "seb", "csv", "answer"], type=str, nargs=1,
                        dest="gold_format",
                        help="Format of the gold annotations. original: path to the original dataset directory to use."
                             "seb: a file that contains the paths of the original dataset files included in a split. "
                             "csv: a file that contains the dataset as a dataframe", required=False, default=["csv"])