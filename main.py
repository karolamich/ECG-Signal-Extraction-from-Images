import argparse
import os
from basic_version import run_basic_version
from extended_version import run_extended_version

def main():
    parser = argparse.ArgumentParser(description="Projekt -Ekstrakcja sygnału EKG")
    parser.add_argument("--input_dir", required=True, help="Ścieżka do folderu z obrazami wejściowymi")
    parser.add_argument("--output_dir", required=True, help="Ścieżka do folderu wyjściowego na pliki CSV")
    parser.add_argument("--mode", choices=['basic', 'scan'], default='basic', help="Tryb działania programu: 'basic' dla wersji podstawowej, 'scan' wersji podstawowej")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'basic':
        print("Wersji podstawowa")
        run_basic_version(args.input_dir, args.output_dir)
        
    elif args.mode == 'scan':
        print("Wersja rozszerzona")
        run_extended_version(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()