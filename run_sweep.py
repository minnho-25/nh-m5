import argparse

from src.experiments import run_sweep


def main():
	parser = argparse.ArgumentParser(description="HuiChain parameter sweep")
	parser.add_argument("--seeds", type=int, default=5)
	parser.add_argument("--output", default="results")
	args = parser.parse_args()
	raw, aggregated = run_sweep(seeds=range(args.seeds), output_dir=args.output)
	print(f"raw runs: {len(raw)}")
	print(f"aggregated groups: {len(aggregated)}")
	print(f"saved: {args.output}/raw/simulation_results.csv")
	print(f"saved: {args.output}/aggregated/experiment_summary.csv")


if __name__ == "__main__":
	main()
