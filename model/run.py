#!/usr/bin/env python3
import sys
import argparse
import pandas as pd
import json
import pickle
import traceback
from pathlib import Path
from typing import Optional, List


def candidate_dirs() -> List[Path]:
    dirs: List[Path] = []
    try:
        dirs.append(Path(__file__).resolve().parent)
    except Exception:
        pass

    try:
        argv0_dir = Path(sys.argv[0]).resolve().parent
        if argv0_dir not in dirs:
            dirs.append(argv0_dir)
    except Exception:
        pass

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        try:
            meipass_path = Path(meipass)
            if meipass_path not in dirs:
                dirs.insert(0, meipass_path)
        except Exception:
            pass

    cwd = Path.cwd()
    if cwd not in dirs:
        dirs.append(cwd)

    return dirs


def find_model_file(candidate_names: List[str]) -> Optional[Path]:
    dirs = candidate_dirs()
    for d in dirs:
        for name in candidate_names:
            p = d.joinpath(name)
            if p.exists():
                return p
    return None


def prepare_model_path(cli_model: Optional[str]):
    tried = []
    if cli_model:
        p = Path(cli_model)
        tried.append(str(p))
        if p.exists():
            return (p, tried)
        for base in candidate_dirs():
            candidate = base.joinpath(cli_model)
            tried.append(str(candidate))
            if candidate.exists():
                return (candidate, tried)

    candidates = ["model/RF_binary_v1.pickle", "RF_binary_v1.pickle"]
    found = find_model_file(candidates)
    if found:
        return (found, [str(found)])

    dirs = candidate_dirs()
    for d in dirs:
        for name in candidates:
            tried.append(str(Path(d).joinpath(name)))
    # default return first candidate as Path (may not exist) plus tried list for debugging
    return (Path(candidates[0]), tried)


def load_model_from_path(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"模型文件不存在: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    try:
        parser = argparse.ArgumentParser(description="Run model prediction (JSON output).")
        parser.add_argument("csv", help="Input CSV file path")
        parser.add_argument(
            "--model",
            "-m",
            help="Optional path to model pickle (overrides bundled model)",
            default=None,
        )
        args = parser.parse_args()

        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"输入 CSV 未找到: {csv_path}")

        model_path, tried_paths = prepare_model_path(args.model)

        # load csv
        df = pd.read_csv(str(csv_path))
        df = df.dropna()
        if df.shape[0] == 0:
            raise ValueError("No rows after dropna()")

        # load model
        try:
            model = load_model_from_path(model_path)
        except Exception as load_exc:
            raise FileNotFoundError(
                f"无法加载模型: attempted paths: {tried_paths}. Exception: {load_exc}"
            )

        # predict (use first row like old script)
        preds = model.predict(df)
        # mapping as in old code
        mapping = {0: "BPA", 1: "UPA"}
        label = mapping[preds[0]]

        # JSON output (simple)
        out = {"success": True, "results": label}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        sys.exit(0)

    except Exception as e:
        tb = traceback.format_exc()
        err_obj = {"success": False, "error": str(e), "traceback": tb}
        # 把错误写 stderr（Electron 仍可捕获并解析）
        sys.stderr.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
        sys.stderr.flush()
        sys.exit(2)


if __name__ == "__main__":
    main()
